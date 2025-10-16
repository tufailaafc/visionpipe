import sys
sys.path.append("../")
import gi
#import configparser
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib
from gi.repository import GstRtspServer
# from ctypes import *
#import time
import sys
import math
import datetime
# import pyds
#import platform




#EOS = End Of Stream
#PGIE = primary generative inference engine


# This will add an rtsp source to the source bin which will allow it to be played
def add_rtsp_source(rtsp_url):
    global source_index
    # This will help to ensure no memory or other errors by having too many streams
    if source_index >= MAX_NUM_SOURCES:
        print("Max number of sources reached.")
        return

    print(f"Adding source #{source_index}: {rtsp_url}")
    source_bin = create_uridecode_bin(source_index, rtsp_url)
    if not source_bin:
        print(f"Failed to create source bin for {rtsp_url}")
        return

    pipeline.add(source_bin)
    g_source_bin_list[source_index] = source_bin

    # Set to PLAYING
    source_bin.set_state(Gst.State.PLAYING)

    g_source_enabled[source_index] = True
    # Keep track of how many sources we have.
    g_num_sources += 1
    source_index += 1


# Configuration constants
#STREAM_CONTAINER="pig-sorting-streamlit" 
MAX_DISPLAY_LEN=64
# Our classes for detection
PGIE_CLASS_ID_STRAWBERRY_IMMATURE = 0
PGIE_CLASS_ID_STRAWBERRY_RIPE = 1
PGIE_CLASS_ID_STRAWBERRY_ROTTEN = 2


# This will set the output dimensions and will scale it up or down to fit.
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 33000

# This will set the dimensions when we have to output more than one stream
TILED_OUTPUT_WIDTH=1280
TILED_OUTPUT_HEIGHT=720

# This is the ID of the host gpu
GPU_ID = 0
MAX_NUM_SOURCES = 1
#SINK_ELEMENT = "nveglglessink"

#This points to the configuration file for the primary inference(model)
PGIE_CONFIG_FILE = "deepstream/config_infer_primary_yolo11.txt"

CONFIG_GPU_ID = "gpu-id"
CONFIG_GROUP_TRACKER = ""

# Global variables
#Contains the number of sources currently
g_num_sources = 0 
#Contains the id for all of the sources
g_source_id_list = [0]*MAX_NUM_SOURCES
#Contains a list that will store which streams have finished
g_eos_list = [False] * MAX_NUM_SOURCES
#Contains a list of all of the sources that we have enabled
g_source_enabled = [False] * MAX_NUM_SOURCES
g_source_bin_list = [None] * MAX_NUM_SOURCES


#The list of classes that are model is configured to detect
pgie_classes_str=['immature strawberry', 'ripe strawberry', 'rotten']

uri = ""
source_index = 0

loop = None
pipeline = None
streammux = None
sink = None
pgie = None
nvvideoconvert = None
nvosd = None
tiler = None
tracker = None

##############################
# Debug functions
# Will print out the details for the rtsp stream like dimensions and memory type, also stops it from breaking, have found out why yet.
print_flag= False
def new_pad_probe(pad, info, user_data):
    caps = pad.get_current_caps()
    if print_flag:
        if caps:
            print("🔍 CAPS DEBUG:", caps.to_string())
        else:
            print("⚠️ No caps found on pad")
    return Gst.PadProbeReturn.OK

infer_frame_count = {}

def pgie_frame_probe(pad, info, user_data):
    """Probe that prints a message every frame arriving at the inference element."""
    pad_name = pad.get_name()

    if pad_name not in infer_frame_count:
        infer_frame_count[pad_name] = 0

    infer_frame_count[pad_name] += 1

    N = 50
    if infer_frame_count[pad_name] % N == 0:
        buf = info.get_buffer()
        if buf:
            pts = buf.pts
            dts = buf.dts
            size = buf.get_size()
            print(f"[PGIE DEBUG] Frame {infer_frame_count[pad_name]} at pad {pad_name}, PTS={pts}, DTS={dts}, Size={size}")

    return Gst.PadProbeReturn.OK


# def pgie_detection_probe(pad, info, u_data):
#     buffer = info.get_buffer()
#     if not buffer:
#         return Gst.PadProbeReturn.OK

#     # Retrieve batch metadata from the buffer
#     batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buffer))
#     l_frame = batch_meta.frame_meta_list

#     while l_frame is not None:
#         frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
#         obj_meta_list = frame_meta.obj_meta_list
#         while obj_meta_list is not None:
#             obj_meta = pyds.NvDsObjectMeta.cast(obj_meta_list.data)
#             print(f"Detected object class id: {obj_meta.class_id}")
#             obj_meta_list = obj_meta_list.next
#         l_frame = l_frame.next

#     return Gst.PadProbeReturn.OK


frame_counter = {}

def pad_debug_probe_limited(pad, info, user_data):
    """Prints pad debug info every N frames."""
    N = 80
    pad_name = pad.get_name()

    if pad_name not in frame_counter:
        frame_counter[pad_name] = 0

    frame_counter[pad_name] += 1

    buf = info.get_buffer()
    if buf and frame_counter[pad_name] % N == 0:
        try:
            pts = buf.pts
            dts = buf.dts
            size = buf.get_size()
            print(f"[PAD DEBUG] Pad: {pad_name}, PTS: {pts}, DTS: {dts}, Size: {size}")

            caps = pad.get_current_caps()
            if caps:
                print(f"[PAD DEBUG] Caps: {caps.to_string()}")
        except Exception as e:
            print(f"Pad debug failed: {e}")

    return Gst.PadProbeReturn.OK


# Lets us know which elements are created and how they are linked
def print_pipeline_status(pipeline):
    """Prints each element and its pads, and checks linking status."""
    print("Pipeline Elements and Pads:")

    for elem in pipeline.iterate_elements():
        print(f"Element: {elem.get_name()} ({elem.get_factory().get_name()})")
        for pad in elem.iterate_pads():
            pad_name = pad.get_name()
            pad_caps = pad.get_current_caps()
            peer_pad = pad.get_peer()
            peer_name = peer_pad.get_name() if peer_pad else "None"
            print(f"  Pad: {pad_name}, Peer: {peer_name}, Caps: {pad_caps.to_string() if pad_caps else 'None'}")

    print("\nChecking link status:")
    for elem in pipeline.iterate_elements():
        for pad in elem.iterate_pads():
            peer_pad = pad.get_peer()
            if peer_pad:
                print(f"{elem.get_name()}:{pad.get_name()} -> {peer_pad.get_parent_element().get_name()}:{peer_pad.get_name()}")
            else:
                print(f"{elem.get_name()}:{pad.get_name()} -> Not linked")

##############################################





def decodebin_child_added(child_proxy, Object, name: str, user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added", decodebin_child_added, user_data)

    if(name.find("nvv4l2decoder") != -1):
        Object.set_property("drop-frame-interval", 0)
        Object.set_property("num-extra-surfaces", 0)

        # Attach probe to decoder's src pad for debugging rtsp input stream, also stops it from breaking for some reason.
        decoder_src_pad = Object.get_static_pad("src")
        if decoder_src_pad:
            print("🧷 Attaching pad probe to decoder's src pad")
            decoder_src_pad.add_probe(Gst.PadProbeType.BUFFER, new_pad_probe, None)

            

def cb_newpad(decodebin, pad, data):
    global streammux
    print("In cb_newpad\n")
    caps = pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()

    print("gstname=", gstname)

    if(gstname.find("video") != -1):
        source_id = data
        pad_name = "sink_%u" % source_id
        print(f"Requesting pad: {pad_name}")
        
        sinkpad = streammux.request_pad_simple(pad_name)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
            return
            
        link_ret = pad.link(sinkpad)
        if link_ret == Gst.PadLinkReturn.OK:
            print("✅ Decodebin linked to pipeline")
        else:
            sys.stderr.write(f"❌ Failed to link decodebin to pipeline: {link_ret}\n")


#This will create a bin that contains our sources
def create_uridecode_bin(index, filename):
    global g_source_id_list
    print("Creating uridecodebin for [%s]" % filename)

    g_source_id_list[index] = index
    bin_name = "source-bin-%02d" % index
    print(bin_name)

    bin = Gst.ElementFactory.make("uridecodebin", bin_name)
    if not bin:
        sys.stderr.write("Unable to create uri decode bin \n")
        return None

    bin.set_property("uri", filename)
    bin.connect("pad-added", cb_newpad, g_source_id_list[index])
    bin.connect("child-added", decodebin_child_added, g_source_id_list[index])

    g_source_enabled[index] = True
    return bin


# Properly release a source stream
def stop_release_source(source_id):
    global g_num_sources
    global g_source_bin_list
    global streammux
    global pipeline

    state_return = g_source_bin_list[source_id].set_state(Gst.State.NULL)

    if state_return == Gst.StateChangeReturn.SUCCESS:
        print("STATE CHANGE SUCCESS\n")
        pad_name = "sink_%u" % source_id
        print(pad_name)

        sinkpad = streammux.get_static_pad(pad_name)
        sinkpad.send_event(Gst.Event.new_flush_stop(False))
        streammux.release_request_pad(sinkpad)
        print("STATE CHANGE SUCCESS\n")

        pipeline.remove(g_source_bin_list[source_id])
        source_id -= 1
        g_num_sources -= 1

    elif state_return == Gst.StateChangeReturn.FAILURE:
        print("STATE CHANGE FAILURE\n")

    elif state_return == Gst.StateChangeReturn.ASYNC:
        state__return = g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)
        pad_name = "sink_%u" % source_id
        print(pad_name)
        sinkpad = streammux.get_static_pad(pad_name)
        sinkpad.send_event(Gst.Event.new_flush_stop(False))
        streammux.release_request_pad(sinkpad)
        print("STATE CHANGE ASYNC\n")
        pipeline.remove(g_source_bin_list[source_id])
        source_id -= 1
        g_num_sources -= 1

def delete_sources(data):
    global loop
    global g_num_sources
    global g_eos_list
    global g_source_enabled

    # Delete sources that have reached end of stream
    for source_id in range(MAX_NUM_SOURCES):
        if (g_eos_list[source_id] and g_source_enabled[source_id]):
            g_source_enabled[source_id] = False
            stop_release_source(source_id)

    # If there are no more sources then quit
    if (g_num_sources == 0):
        loop.quit()
        print("All sources stopped quitting")
        return False


# Properly handles adding sources
def add_sources(data):
    global g_num_sources
    global g_source_enabled
    global g_source_bin_list
    global pipeline

    source_id = g_num_sources

    # Update our list so that we know this source is enabled
    g_source_enabled[source_id] = True
    print("Calling Start %d " % source_id)

    # create a bin that contains our source
    source_bin = create_uridecode_bin(source_id, uri)
    if (not source_bin):
        sys.stderr.write("Failed to create bin. Exiting.")
        exit(1)

    # Add our bin to th global variable to manage
    g_source_bin_list[source_id] = source_bin
    # Add the bin to the actual pipeline
    pipeline.add(source_bin)

    # Enable our source so that it can start streaming
    state_return = g_source_bin_list[source_id].set_state(Gst.State.PLAYING)


    # Evaluate if it was successful
    if state_return == Gst.StateChangeReturn.SUCCESS:
        print("STATE CHANGE SUCCESS\n")
        source_id += 1
    elif state_return == Gst.StateChangeReturn.FAILURE:
        print("STATE CHANGE FAILURE\n")
    elif state_return == Gst.StateChangeReturn.ASYNC:
        state_return = g_source_bin_list[source_id].get_state(Gst.CLOCK_TIME_NONE)
        source_id += 1
    elif state_return == Gst.StateChangeReturn.NO_PREROLL:
        print("STATE CHANGE NO PREROLL\n")

    g_num_sources += 1

    #To-Do add some proper logic to prevent adding to many sources
    if (g_num_sources == MAX_NUM_SOURCES):
        print(f"Over max number of sources {g_num_sources}")

    return True



# This will handle messages and make descions based off of that.
def bus_call(bus, message, loop):
    global g_eos_list

    t = message.type

    # If we recieve and end of stream we will quit
    if t == Gst.MessageType.EOS:
        sys.stdout.write("End-of-stream\n")
        loop.quit()

    elif t == Gst.MessageType.WARNING:
        err, debug = message.parse_warning()
        sys.stderr.write("Warning: %s: %s: \n" % (err, debug))

    # If we recieve a error we will also quit
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        sys.stderr.write("Error: %s: %s: \n" % (err, debug))
        loop.quit()

    elif t == Gst.MessageType.ELEMENT:
        struct = message.get_structure()
        # We add the stream that ended to our list to handle stream ending gracefully
        if struct is not None and struct.has_name("stream-eos"):
            parsed, stream_id = struct.get_uint("stream-id")
            if parsed:
                print("Got EOS from stream %d" % stream_id)
                g_eos_list[stream_id] = True
        elif struct.has_name("perf"):
            print("PERF EVENT: ", struct.to_string())
    return True

def main(args):
    global source_index
    global g_num_sources
    global g_source_bin_list
    global uri
    global loop
    global pipeline
    global streammux
    global sink
    global pgie
    global nvvideoconvert
    global nvosd
    global tiler
    global tracker

    source_index = 0

    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <uri1> \n" % args[0])
        sys.exit(1)

    num_sources = len(args) - 1

    # Standard GStreamer initialization
    Gst.init(None)
    # Set logging levels
    Gst.debug_set_active(True)
    Gst.debug_set_default_threshold(3)

    # Create Pipeline elementis_live
    print("Creating Pipeline \n")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write("Unable to create pipeline \n")
        sys.exit(1)

    # Create nvstreammux to form batches from one or more sources
    print("Creating streammux \n")
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write("Unable to create NvStreamMux \n")
        sys.exit(1)

    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("batch-size", 1)
    streammux.set_property("gpu_id", GPU_ID)
    streammux.set_property("live-source", 1)
    streammux.set_property("width", MUXER_OUTPUT_WIDTH)
    streammux.set_property("height", MUXER_OUTPUT_HEIGHT)

    pipeline.add(streammux)

    # add sources from command line
    uri = args[1]
    for i in range(num_sources):
        print("Creating source_bin ", i, " \n")
        uri_name = args[i+1]
        if uri_name.find("rtsp://") == 0:
            is_live = True

        source_bin = create_uridecode_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Failed to create source bin, Exiting. \n")
            sys.exit(1)

        g_source_bin_list[i] = source_bin
        pipeline.add(source_bin)

    g_num_sources = num_sources

    # Create primary inference engine, this will determine if there are any instances of the class in the incoming source
    print("Creating Pgie \n")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write("Unable to create Pgie \n")
        sys.exit(1)
    ########################################
    # pgie_sink_pad = pgie.get_static_pad("src")  # or "sink", see below
    # if not pgie_sink_pad:
    #     sys.stderr.write("Unable to get src pad of PGIE\n")
    # else:
    #     pgie_sink_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_detection_probe, None)
    ########################################


    # Create tracker, this will track the object as it moves in the video stream
    print("Creating nvtracker \n")
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write("Unable to create nvtracker \n")
        sys.exit(1)

    # Create tiler, this will configure the 2d tile for new sources being added
    print("Creating tiler \n")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write("Unable to create tiler \n")
        sys.exit(1)

    # Create nvvideoconvert, will do scaling  cropping and video color format conversion
    print("Creating nvvidconv \n")
    nvvideoconvert = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvideoconvert:
        sys.stderr.write("Unable to create nvvidconv \n")
        sys.exit(1)

    # Create nvdsosd, this will handle drawing bounding boxes around the region of intrest
    print("Creating nvosd \n")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write("Unable to create nvosd \n")
        sys.exit(1)

    # Create tee for splitting output, so we can save to file and display the output
    print("Creating tee \n")
    tee = Gst.ElementFactory.make("tee", "tee")
    if not tee:
        sys.stderr.write("Unable to create tee \n")
        sys.exit(1)

    # Create queues
    queue_display = Gst.ElementFactory.make("queue", "queue-display")
    queue_file = Gst.ElementFactory.make("queue", "queue-file")
    if not queue_display or not queue_file:
        sys.stderr.write("Unable to create queues \n")
        sys.exit(1)

    # Create display sink (fakesink for now)
    display_sink = Gst.ElementFactory.make("fakesink", "display-sink")
    if not display_sink:
        sys.stderr.write("Unable to create display sink \n")
        sys.exit(1)
    display_sink.set_property("sync", False)


    ################################ File output elements########################################
    #Handles scaling cropping etc
    print("Creating nvvidconv2 \n")
    nvvideoconvert2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    if not nvvideoconvert2:
        sys.stderr.write("Unable to create nvvidconv2 \n")
        sys.exit(1)
    

    #Ensures that the stream is in the correct format
    print("Creating capsfilter \n")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    if not capsfilter:
        sys.stderr.write("Unable to create capsfilter \n")
        sys.exit(1)
    # Convert NVMM to system memory for encoder compatibility
    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    # Encodes it into a format that can be saved into a file
    print("Creating encoder (x264enc) \n")
    encoder = Gst.ElementFactory.make("x264enc", "encoder")
    if not encoder:
        # Fallback to software encoder if x264enc not available
        print("x264enc not available, trying avenc_h264")
        encoder = Gst.ElementFactory.make("avenc_h264", "encoder")
        if not encoder:
            sys.stderr.write("Unable to create any H.264 encoder \n")
            sys.exit(1)
    
    # Set encoder properties for better performance
    if encoder.get_factory().get_name() == "x264enc":
        encoder.set_property("bitrate", 2000)
        encoder.set_property("tune", "zerolatency")
        encoder.set_property("speed-preset", "ultrafast")

    else:
        encoder.set_property("bitrate", 2000000)

    print("Creating code parser (h264parse) \n")
    codeparser = Gst.ElementFactory.make("h264parse", "h264-parse")
    if not codeparser:
        sys.stderr.write("Unable to create h264parse \n")
        sys.exit(1)

    print("Creating container \n")
    container = Gst.ElementFactory.make("qtmux", "qtmux")
    if not container:
        sys.stderr.write("Unable to create container \n")
        sys.exit(1)

    print("Creating file sink \n")
    file_sink = Gst.ElementFactory.make("filesink", "filesink")
    if not file_sink:
        sys.stderr.write("Unable to create file sink \n")
        sys.exit(1)
    # This will set where we output our file to, and what we call it.
    timestamp = datetime.datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    file_sink.set_property("location", f"/app/videos/deepstream_{timestamp}_out.mp4")
    file_sink.set_property("sync", 1)
    file_sink.set_property("async", 0)

    if is_live:
        print("At least one of the sources is live.")
        streammux.set_property("live-source", 1)

    # Set pgie properties
    pgie.set_property("config-file-path", PGIE_CONFIG_FILE)
    pgie_batch_size = pgie.get_property("batch-size")
    if(pgie_batch_size < MAX_NUM_SOURCES):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ", num_sources, "\n")
    pgie.set_property("batch-size", MAX_NUM_SOURCES)
    pgie.set_property("gpu_id", GPU_ID)

    # Set tiler properties
    tiler_rows = int(math.sqrt(num_sources))
    tiler_columns = int(math.ceil((1.0*num_sources)/tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    # Set gpu IDs
    tiler.set_property("gpu_id", GPU_ID)
    nvvideoconvert.set_property("gpu_id", GPU_ID)
    nvosd.set_property("gpu_id", GPU_ID)
    nvvideoconvert2.set_property("gpu_id", GPU_ID)
    

    print("Adding elements to Pipeline \n")
    elements = [pgie, tiler, nvvideoconvert, nvosd, tee, queue_display, queue_file,
                display_sink, nvvideoconvert2, capsfilter, encoder, codeparser, container, file_sink]
    
    for element in elements:
        pipeline.add(element)

    # Link main pipeline, this determines how the order in which the information flows
    print("Linking elements in the pipeline \n")
    
    if not streammux.link(pgie):
        sys.stderr.write("Failed to link streammux -> pgie\n")
        sys.exit(1)
        
    if not pgie.link(tiler):
        sys.stderr.write("Failed to link pgie -> tiler\n")
        sys.exit(1)
        
    if not tiler.link(nvvideoconvert):
        sys.stderr.write("Failed to link tiler -> nvvideoconvert\n")
        sys.exit(1)
        
    if not nvvideoconvert.link(nvosd):
        sys.stderr.write("Failed to link nvvideoconvert -> nvosd\n")
        sys.exit(1)

    if not nvosd.link(tee):
        sys.stderr.write("Failed to link nvosd -> tee\n")
        sys.exit(1)

    # Branch 1: Display
    tee_src_pad_display = tee.get_request_pad("src_%u")
    queue_display_sink_pad = queue_display.get_static_pad("sink")
    if tee_src_pad_display.link(queue_display_sink_pad) != Gst.PadLinkReturn.OK:
        sys.stderr.write("Failed to link tee -> queue_display\n")
        sys.exit(1)
    
    if not queue_display.link(display_sink):
        sys.stderr.write("Failed to link queue_display -> display_sink\n")
        sys.exit(1)

    # Branch 2: File output
    tee_src_pad_file = tee.get_request_pad("src_%u")
    queue_file_sink_pad = queue_file.get_static_pad("sink")
    if tee_src_pad_file.link(queue_file_sink_pad) != Gst.PadLinkReturn.OK:
        sys.stderr.write("Failed to link tee -> queue_file\n")
        sys.exit(1)
    
    if not queue_file.link(nvvideoconvert2):
        sys.stderr.write("Failed to link queue_file -> nvvideoconvert2\n")
        sys.exit(1)
        
    if not nvvideoconvert2.link(capsfilter):
        sys.stderr.write("Failed to link nvvideoconvert2 -> capsfilter\n")
        sys.exit(1)
        
    if not capsfilter.link(encoder):
        sys.stderr.write("Failed to link capsfilter -> encoder\n")
        sys.exit(1)
        
    if not encoder.link(codeparser):
        sys.stderr.write("Failed to link encoder -> codeparser\n")
        sys.exit(1)

    # Link to container using request pad
    sinkpad_video = container.get_request_pad("video_0")
    if not sinkpad_video:
        sys.stderr.write("Unable to get video sink pad from qtmux\n")
        sys.exit(1)
        
    srcpad_parser = codeparser.get_static_pad("src")
    if not srcpad_parser:
        sys.stderr.write("Unable to get src pad from parser\n")
        sys.exit(1)
        
    if srcpad_parser.link(sinkpad_video) != Gst.PadLinkReturn.OK:
        sys.stderr.write("Failed to link parser -> qtmux\n")
        sys.exit(1)

    if not container.link(file_sink):
        sys.stderr.write("Failed to link qtmux -> filesink\n")
        sys.exit(1)




    # Create event loop and bus
    #Maind logic loop for pipeline
    loop = GLib.MainLoop()

    # Where we manage messages from pipeline
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write("Unable to get the sinkpad of nvosd \n")

    # Set source bins to PLAYING state
    print("Setting source bins to PLAYING")
    for i in range(num_sources):
        if g_source_bin_list[i]:
            ret = g_source_bin_list[i].set_state(Gst.State.PLAYING)
            print(f"Source bin {i} state change: {ret}")

    # Set pipeline to PAUSED first
    print("Setting pipeline to PAUSED")
    ret = pipeline.set_state(Gst.State.PAUSED)
    if ret == Gst.StateChangeReturn.FAILURE:
        sys.stderr.write("Unable to set pipeline to PAUSED\n")
        sys.exit(1)

    # Wait for PAUSED state
    ret, state, pending = pipeline.get_state(10 * Gst.SECOND)
    if ret == Gst.StateChangeReturn.FAILURE:
        sys.stderr.write("Failed to get pipeline state\n")
        sys.exit(1)
    
    print(f"Pipeline PAUSED state: {state}, pending: {pending}")

    # Sync new elements with parent
    for element in [tee, queue_display, queue_file, display_sink]:
        if element:
            element.sync_state_with_parent()

    # Now set to PLAYING
    print("Setting pipeline to PLAYING")
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        sys.stderr.write("Unable to set pipeline to PLAYING\n")
        sys.exit(1)

    # Wait for PLAYING state
    print("Waiting for state change to PLAYING")
    ret, state, pending = pipeline.get_state(10 * Gst.SECOND)
    print(f"Final pipeline state: {state}, pending: {pending}, status: {ret}")
    
    if state == Gst.State.PLAYING and pending == Gst.State.VOID_PENDING:
        print("✅ Pipeline is successfully PLAYING")
        # Display the data sources
        print("Now playing...")
        for i, source in enumerate(args):
            if(i != 0):
                print(i, ": ", source)
        print_pipeline_status(pipeline)
    else:
        print("❌ Pipeline failed to reach PLAYING state")
        print_pipeline_status(pipeline)
        sys.exit(1)

    try:
        loop.run()
    except:
        pass

    # Cleanup
    print("Exiting app \n")
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    sys.exit(main(sys.argv))