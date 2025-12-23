#constants
LOG_DATA_PKL = "data.pkl"
LOG_MODEL_PKL = "model.pkl"
LOG_METRIC_PKL = "metrics.model"

#libraries
import os 
import sklearn
import pickle
import yaml
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


class JobPrediction:
    """Production Class for predicting the probability of a job from skills"""

    #constructor
    def __init__(self,mlflow_uri , run_id ,exp_id, clusters_yaml_path):
        
        #constants
        self.tracking_uri = mlflow_uri
        self.run_id = run_id
        self.exp_id = exp_id

        #retrieve model and features
        mlflow_objs = self.load_mlflow_objs()
        self.model = mlflow_objs[0]
        self.features_names = mlflow_objs[1]
        self.target_names = mlflow_objs[2]

        #load cluster config
        self.path_clusters_config = clusters_yaml_path
        self.skills_clusters_df = self.load_clusters_config(clusters_yaml_path)

    def load_mlflow_objs(self):
        """"Load objects from the MLflow run"""
        mlflow.set_tracking_uri(self.tracking_uri)
        client = MlflowClient()

        run = mlflow.get_run(self.run_id)

        #load model
        model_path = os.path.join(
            self.tracking_uri, 
            self.exp_id, 
            self.run_id, 
            "artifacts", 
            LOG_MODEL_PKL
        )

        with open(model_path, "rb") as f:
            model_pkl = pickle.load(f)

        #load data
        data_path = os.path.join(
            self.tracking_uri, 
            self.exp_id, 
            self.run_id, 
            "artifacts", 
            LOG_DATA_PKL
        )
        with open(data_path, "rb") as f:
            data_pkl = pickle.load(f)

        #return model and data labels
        return model_pkl['model_object'],data_pkl["features_names"],data_pkl["targets_names"]
    
    def load_clusters_config(self, path_cluster_config):
        """Load skills clusters developed in 03_feature_engineering.ipynb"""

        #Read yaml
        with open(path_cluster_config ,'r') as stream:
            clusters_config = yaml.safe_load(stream)

        molten_clusters = [
            (cluster_name , cluster_skill) 
            for cluster_name , cluster_skills in clusters_config.items()
            for cluster_skill in cluster_skills
        ]

        clusters_df = pd.DataFrame(molten_clusters , columns = ['cluster_name' , 'skill'])

        return clusters_df
    
    #GETTERS

    def get_all_skills(self):
        return self.features_names
    
    def get_all_jobs(self):
        return self.target_names
    
    #Predictions
    def create_features_array(self, available_skills):
        """Create the features array from a list of the available skills"""
        def create_clusters_features(self , available_skills):
            sample_clusters = self.skills_clusters_df.copy()
            sample_clusters['available_skills'] = sample_clusters['skill'].isin(available_skills)
            cluster_features = sample_clusters.groupby('cluster_name')['available_skills'].sum()

            return cluster_features
        

        def create_skills_features(self,available_skills , exclude_features):
            all_features = pd.Series(self.features_names.copy())
            skills_names = all_features[~all_features.isin(exclude_features)]
            ohe_skills = pd.Series(skills_names.isin(available_skills).astype(int).tolist(), index=skills_names)

            return ohe_skills
        
        clusters_features = create_clusters_features(self , available_skills)
        skills_features = create_skills_features(self , available_skills ,clusters_features.index)

        #combine features and sort
        features = pd.concat([skills_features , clusters_features])
        features = features[self.features_names]

        return features.values
    
    def predict_jobs_probs(self, available_skills):
        """Returns probabilities of the different jobs according to the skills"""

        features_array = self.create_features_array(available_skills)

        predictions = self.model.predict_proba([features_array])
        predictions =[prob[0][1] for prob in predictions]
        predictions = pd.Series(predictions , index = self.target_names)

        return predictions
    

    ##simulation

    def recommend_new_skills(self , available_skills ,target_job ,threshold=0):
        
        #calc base prob
        base_pred = self.predict_jobs_probs(available_skills)

        #get all possible additional skills
        all_skills = pd.Series(self.get_all_skills())
        new_skills = all_skills[~all_skills.isin(available_skills)].copy()

        #simulate new skills
        simulated_res = []
        for skill in new_skills:
            additional_skill_prob = self.predict_jobs_probs([skill] + available_skills)
            additional_skill_uplift = (additional_skill_prob - base_pred)/base_pred
            additional_skill_uplift.name = skill
            simulated_res.append(additional_skill_uplift)

        simulated_res = pd.DataFrame(simulated_res)

        #recommend new skills
        target_res = simulated_res[target_job].sort_values(ascending=False)
        pos_mask = (target_res >threshold)
        return target_res[pos_mask]



