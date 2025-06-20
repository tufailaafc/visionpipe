Python Requirements: pymongo, opencv.
Requires opencv-python==4.10.0.84 newer versions will cause timeout error. pip install opencv-python==4.10.0.84



Setting up MongoDB in WSL2

Requirements: WSL2, Ubuntu 22.04.

Step 1: Run the command: "sudo apt-get install gnupg curl" in the wsl2 terminal.

Step 2: Run the command: "curl -fsSL https://www.mongodb.org/static/pgp/server-8.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-8.0.gpg \
   --dearmor" This will get MongoDB public key.

Step 3: Run "cat /etc/lsb-release" to see what version of ubuntu you have.

Step 4: If running Jammy use the command "echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-8.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/8.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-8.0.list" to create the list file.

Step 4: Reload the package database "sudo apt-get update"

Step 5: Install latest release of monogoDB Community Server "sudo apt-get install -y mongodb-org"

Step 5.5: May have to run "sudo gitlab-ctl reconfigure" if you have issues with GitLabee.

Step 6: If you run "mongod" at this point you will likely get this error "Data directory /data/db not found. Create the missing directory or specify another path using (1) the --dbpath command line option" this is because we are running this in WSL and it created the folder in "~/" not "/".

Step 7: Move to the root "cd /"

Step 8: Create the directory "sudo mkdir -p data/db"

Step 9: If you run "mongod" now you may now have a different error. This is because it needs permissions.

Step 10: Run: "sudo chown -R `id -un` data/db" to grant it permissions.

Step 11: Run: "mongod" which will start up the server and then run "mongosh" in another terminal to connect and verify that it is running.