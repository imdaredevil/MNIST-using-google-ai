from google.api_core.client_options import ClientOptions
from googleapiclient import discovery
from utils import prepare_test_data, convert_to_list

ENDPOINT = 'https://us-central1-ml.googleapis.com'
PROJECT = 'marine-album-313805'
MODEL_NAME = 'mnist_prediction'
BATCH_SIZE = 10

client_options = ClientOptions(api_endpoint=ENDPOINT)
service = discovery.build('ml', 'v1', client_options=client_options)
MODEL_PATH = 'projects/{}/models/{}'.format(PROJECT, MODEL_NAME)


def predictJson(instances, version=None):
    """Send json data to a deployed model for prediction.

    Args:
        project (str): project where the AI Platform Model is deployed.
        model (str): model name.
        instances ([Mapping[str: Any]]): Keys should be the names of Tensors
            your deployed model expects as inputs. Values should be datatypes
            convertible to Tensors, or (potentially nested) lists of datatypes
            convertible to tensors.
        version: str, version of the model to target.
    Returns:
        Mapping[str: any]: dictionary of prediction results defined by the
            model.
    """
    model_path = MODEL_PATH
    if version is not None:
        model_path += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=model_path,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']

def makeOnlinePredictions():
    batch_size = BATCH_SIZE
    x = prepare_test_data('./dataset/test.csv')
    xlist = convert_to_list(x, 50)
    numRecords = len(xlist)
    predictions = {}
    i = 0
    while i < numRecords:
        try:
            xBatch = xlist[i:]
            if i + batch_size < numRecords:
                xBatch = xlist[i:i + batch_size]
            else:
                batch_size = numRecords - i
            result = predictJson(xBatch)
            resultDict = {j+i : result[j] for j in range(batch_size)}
            predictions.update(resultDict)
        except RuntimeError as error:
            print('The following error occured while processing batch ${0} - ${1}'.format(i, i + BATCH_SIZE))
            print(error)
        i += BATCH_SIZE
    print(predictions)

if __name__ == '__main__':
    makeOnlinePredictions()

    
    
    
