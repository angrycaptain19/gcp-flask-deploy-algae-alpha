# Deploying Flask application on Google Cloud Platform


1. Create GCP Project 

2. Clone this repo into the terminal of your GCP project ensuring the yaml file points to the entrypoint for this project and changing any other deploy configuration you might need. 

3. In GCP terminal 

```bash
   $ cd gcp-flask-deploy-algae-alpha

   $ gcloud app deploy
```

May take a couple of mins for deploy to complete, take note of the web address
given to access the app online once process complete. 


To remove the deployment, type in 'manage resources' in the search bar find 
the project-id and delete. (NOTE: need a cleaner way of removing a deployment) 