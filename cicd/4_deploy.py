import os
import cmlapi
from cmlapi.utils import Cursor
import string
import random
import json

cluster = os.getenv("CDSW_DOMAIN")
client = cmlapi.default_client()

session_id = "".join([random.choice(string.ascii_lowercase) for _ in range(6)])

# List projects using the default sort and default page size (10)
client.list_projects(page_size = 20)

project_id = os.environ["CDSW_PROJECT_ID"]


### CREATE AN ENDPOINT AND PUSH THE  MODEL ###

# Would be nice to name it with job id rather than session id
modelReq = cmlapi.CreateModelRequest(
    name = "pump-model-rf-2",
    description = "Pump predictive mainten Model",
    project_id = project_id,
    disable_authentication = True
)

model = client.create_model(modelReq, project_id)

model_build_request = cmlapi.CreateModelBuildRequest(
    project_id = project_id,
    model_id = model.id,
    comment = "Deplying model as REST api model",
    file_path = "model/model_endpoint.py",
    function_name = "predict",
    kernel = "python3",
    runtime_identifier = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-jupyterlab-python3.8-standard:2022.04.1-b6"
)

modelBuild = client.create_model_build(
    model_build_request, project_id, model.id
)

model_deployment = cmlapi.CreateModelDeploymentRequest(
        project_id = project_id, 
        model_id = model.id, 
        build_id = modelBuild.id, 
        cpu = 1.00,
        memory = 2.00,
        replicas = 4
    )

model_deployment_response = client.create_model_deployment(
        model_deployment, 
        project_id = project_id, 
        model_id = model.id, 
        build_id = modelBuild.id
    )