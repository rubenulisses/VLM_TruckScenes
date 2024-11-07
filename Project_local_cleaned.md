# Model Setup


```python
from huggingface_hub import notebook_login
#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
#dapi34e71f182e6bb4f156b0878992f357d3
# Login to Huggingface to get access to the model
notebook_login()
```


```python
%pip install --upgrade transformers==4.39.0 mlflow>=2.11.3 tensorflow==2.16.1
```


```python
dbutils.library.restartPython()
```


```python
import os
os.environ["HF_HOME"] = "/Volumes/perception-munich/test/llava"
```


```python
model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
```


```python
import numpy as np
from mlflow.models import infer_signature

model_input = np.array(
    [
        ["<Some IMG BYTES 1>",
         "<Some IMG BYTES 2>", 
         "<Some IMG BYTES 3>", 
         "<Some IMG BYTES 4>",
         "<Some IMG BYTES 5>",
         "<Some Instruction>"],  # Adding instruction at the end
        
        ["<Some IMG BYTES 1>", 
         "<Some IMG BYTES 2>", 
         "<Some IMG BYTES 3>", 
         "<Some IMG BYTES 4>",
         "<Some IMG BYTES 5>",
         "<Some Instruction>"]  # Adding instruction at the end
    ]
)

# Sample output data (replace <Sample output> with actual output)
model_output = np.array(
    [
        ["<Sample output 1>"],  # One output for all inputs
        ["<Sample output 2>"]
    ]
)

# Define the signature for the model input and output
signature = infer_signature(model_input, model_output)

print(signature)

```


```python
import mlflow
import numpy as np
import pandas as pd
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from io import BytesIO
import base64
import gc

class Model(mlflow.pyfunc.PythonModel):
    def __init__(self, model_id):
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # Mixed precision to save GPU memory
            low_cpu_mem_usage=True,  # Load model with low CPU memory usage
            cache_dir="/Volumes/perception-munich/test/llava"
        )
        self.model.to("cuda:0")  # Move the model to GPU for computation
        self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing for memory optimization

    def predict(self, context, model_input):
        processor = self.processor
        model = self.model
        results = []

        # Ensure model_input is a list of inputs
        if isinstance(model_input, np.ndarray):
            model_input = model_input.tolist()

        for mi in model_input:
            img_base64_list = mi[:-1]
            prompt = mi[-1]

            # Process images on CPU
            images = []
            for img_base64 in img_base64_list:
                try:
                    image_data = base64.b64decode(img_base64)
                    image = Image.open(BytesIO(image_data)).convert("RGB")
                    images.append(image)
                except Exception as e:
                    return {"error": f"Failed to decode and load image: {str(e)}"}

            # Prepare inputs on CPU and move tensors to GPU only when needed
            input_tensors = processor(prompt, images, return_tensors='pt').to("cuda:0", torch.float16)

            with torch.no_grad():  # Disable gradient calculations to save memory
                output = model.generate(**input_tensors, max_new_tokens=150, do_sample=False)
                
            result = processor.decode(output[0], skip_special_tokens=True)

            if not result.strip():
                print(f"Warning: No description generated for prompt: {prompt}")

            results.append(result)

            # Move tensors back to CPU when done to free GPU memory
            input_tensors.to("cpu")
            del input_tensors, output
            torch.cuda.empty_cache()
            gc.collect()

        return results

```


```python
import os
import shutil
import mlflow

# Define the model directory
model_dir = "/tmp/my_model_dir"

# Check if the directory exists before trying to delete it
if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

mlflow.pyfunc.save_model(
    path=model_dir,
    python_model=Model(model_id=model_id), 
    pip_requirements=[
        'transformers==4.39.0',
        'mlflow==2.11.3',
        'tensorflow',
        'torch',
        'Pillow', 
        'requests'
    ],
    signature=signature 
)

print(f"Model saved to {model_dir}")

```


```python
import mlflow

# Load the model to get its dependencies
model_uri = "/tmp/my_model_dir"  # Replace with your model path
dependencies = mlflow.pyfunc.get_model_dependencies(model_uri)

# Print the dependencies
print(dependencies)
```


```python
%pip install -r /tmp/my_model_dir/requirements.txt
```


```python
dbutils.library.restartPython()
```

# Load Dataset

### Mini Dataset


```python
import zipfile
import os

dbfs_file_path = "dbfs:/FileStore/MAN_TruckScenes_mini_V1.zip"  # Path to your ZIP file in DBFS
local_temp_path = "/tmp/MAN_TruckScenes_mini_V1.zip"  # Temporary local path for the ZIP file

dbutils.fs.cp(dbfs_file_path, "file:/tmp/MAN_TruckScenes_mini_V1.zip")

with zipfile.ZipFile(local_temp_path, 'r') as zip_ref:
    zip_ref.extractall("/tmp/MAN_TruckScenes_mini_V1")  # Unzipping to a local folder
```

### Full Dataset


```python
import zipfile
import os
import shutil

# Paths to the multi-part full_dataset files in DBFS
dbfs_files = [f"dbfs:/FileStore/ezyZip.zip"] + [
    f"dbfs:/FileStore/ezyZip.z{str(i).zfill(2)}" for i in range(1, 19)]

# Local temporary paths
local_dir = "/tmp/full_dataset/"
if os.path.exists(local_dir):
    shutil.rmtree(local_dir)

# Create the directory
os.makedirs(local_dir)

# Copy each part from DBFS to the local temporary directory
for dbfs_file in dbfs_files:
    filename = dbfs_file.split('/')[-1]
    local_file_path = os.path.join(local_dir, filename)
    dbutils.fs.cp(dbfs_file, f"file:{local_file_path}")
```


```python
%sh
# Remove incomplete unzipped files, if any
rm -rf /tmp/full_dataset_unzipped
rm -rf /tmp/full_dataset_final

# Re-concatenate to make sure no corrupt bytes are included
cat /tmp/full_dataset/ezyZip.z* > /tmp/full_dataset/ezyZip_complete.zip

# Attempt extraction
unzip -FF /tmp/full_dataset/ezyZip_complete.zip -d /tmp/full_dataset_unzipped

cd /tmp/full_dataset_unzipped
unzip full_dataset.zip -d /tmp/full_dataset_final
```

# Install Truckscenes


```python
!pip install "truckscenes-devkit[all]"
```


```python
dbutils.library.restartPython()
```


```python
%matplotlib inline
from truckscenes import TruckScenes

# Edit the path to the mini / full dataset
path_to_folder = "/tmp/full_dataset_final/"
trucksc = TruckScenes('v0.1-trainval_LFMC', path_to_folder, True)
```

# Evaluation of the Model

### Get all scenes images


```python
lst_scenes_samples = []
lst_scenes_samples_timestamps = []
lst_scenes_desc = []

for scene in trucksc.scene:
    lst_scenes_desc.append(scene['description'])
    first_sample_token = scene["first_sample_token"]
    sample = trucksc.get("sample", first_sample_token)
    
    lst_samples = []
    lst_scenes_samples_timestamps.append(sample['timestamp'])
    
    # Access 40 consecutive samples (or more if needed)
    for i in range(40):
        
        # Get the sensor data tokens (e.g., camera, lidar, etc.)
        for sensor, token in sample['data'].items():
            # Filter to only process cameras
            if "CAMERA_LEFT_FRONT" in sensor: #remove if you want all sensors
                # Fetch the actual sensor data using the token
                sample_data = trucksc.get("sample_data", token)
                
                lst_samples.append("/tmp/full_dataset_final/" + sample_data['filename'])

        # Move to the next sample (if available)
        if sample['next'] != "":
            sample = trucksc.get("sample", sample['next'])
        else:
            break  # End of scene

    lst_scenes_samples.append(lst_samples)

print(lst_scenes_samples)
```

### Load the Model


```python
import mlflow

# Load the saved model for local testing
loaded_model = mlflow.pyfunc.load_model("/tmp/my_model_dir")
```


```python
import torch
import gc
torch.cuda.empty_cache()
gc.collect()
```

### Get model output


```python
import re

def clean_output(result):
    # Remove everything between [INST] and [/INST] including the tags
    cleaned_result = re.sub(r'\[INST\].*?\[\/INST\]', '', result, flags=re.DOTALL).strip()
    return cleaned_result.strip().strip("'").strip('"')
```


```python
from datetime import datetime
import pytz

def get_daytime_and_season(timestamp):
    # Convert the microseconds timestamp to seconds
    timestamp_seconds = timestamp / 1e6

    # Convert the timestamp to a datetime object in UTC
    dt_utc = datetime.utcfromtimestamp(timestamp_seconds)

    # Define the Germany timezone
    germany_tz = pytz.timezone("Europe/Berlin")

    # Convert UTC datetime to Germany local time
    dt_germany = dt_utc.replace(tzinfo=pytz.utc).astimezone(germany_tz)

    # Determine the daytime based on the hour
    hour = dt_germany.hour
    if 6 <= hour < 11:
        daytime_tag = 'morning'
    elif 11 <= hour < 18:
        daytime_tag = 'noon'
    elif 18 <= hour < 21:
        daytime_tag = 'evening'
    else:
        daytime_tag = 'night'

    # Determine the season based on the month and day
    month = dt_germany.month
    day = dt_germany.day

    if (month == 3 and day >= 20) or (3 < month < 6) or (month == 6 and day <= 20):
        season_tag = 'spring'
    elif (month == 6 and day >= 21) or (6 < month < 9) or (month == 9 and day <= 22):
        season_tag = 'summer'
    elif (month == 9 and day >= 23) or (9 < month < 12) or (month == 12 and day <= 20):
        season_tag = 'autumn'
    else:
        season_tag = 'winter'

    # Return the formatted result
    return f'daytime.{daytime_tag};season.{season_tag}'

```


```python
def add_daytime_and_season(model_output, timestamp):
    # Get the daytime and season from the function
    daytime_season = get_daytime_and_season(timestamp)
    
    # Insert the daytime and season into the model's output
    # Split the output string to insert the daytime and season in the right position
    output_parts = model_output.split(";")
    
    # Insert daytime and season in the right position (between area and lighting)
    new_output = f"{output_parts[0]};{output_parts[1]};{daytime_season};{';'.join(output_parts[2:])}"
    
    return new_output
```


```python
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import torch
import gc
import random
import matplotlib.pyplot as plt

all_results = []
all_selected_images = []

i = 0
# Iterate over your samples list to load images and prepare the input
for scene_samples in lst_scenes_samples:
    # Gather base64 images
    images_base64 = []

    # Calculate 5 equally spaced indices
    num_indices = 5
    sample_indices = [int(i * (len(scene_samples) - 1) / (num_indices - 1)) for i in range(num_indices)]

    # Initialize a list to hold the selected images and timestamps
    selected_images = []

    # Loop through the calculated sample indices
    for sample_idx in sample_indices:
        selected_image = scene_samples[sample_idx]
        selected_images.append(selected_image)

    all_selected_images.append(selected_images)

    # Now, we have selected 5 images and their respective timestamps
    for sample_camera_path in selected_images:
        try:
            # Open the image file directly using the sample_token path
            with Image.open(sample_camera_path) as img:
                buffer = BytesIO()
                # Save the image to the buffer in PNG format
                img.save(buffer, format="PNG")
                buffer.seek(0)  # Go back to the beginning of the buffer
                
                # Convert buffer content to base64
                img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                images_base64.append(img_base64)  # Append each image's base64

        except Exception as e:
            print(f"Error processing {sample_camera_path}: {str(e)}")

    # Single prompt for the scene
    prompt = f"""
        [INST]

        <image> <image> <image> <image> <image> \n

        Based on the 5 images, create a scene description using the categories below. 
        
        The scene description must follow this format strictly:

        'weather.<weather_tag>;area.<area_tag>;lighting.<lighting_tag>;structure.<structure_tag>;construction.<construction_tag>'

        You must replace the tags (<weather_tag>, <area_tag>, <lighting_tag>, <structure_tag>, and <construction_tag>) with one of the respective category option.

        An example of a correct output is: 'weather.rain;area.city;lighting.glare;structure.tunnel;construction.roadworks'

        Tags categories:

        <weather_tag> - Describe the weather conditions, and the options for the <weather_tag> are:
            - clear: The sky is clear, or slightly cloudy and the weather is sunny.  
            - rain: It is raining, either lightly or heavily and the ground might be wet.  
            - snow: The surroundings have snow, or it is snowing.  
            - fog: There is fog affecting visibility.  
            - hail: It is hailing, or the ground is covered in hail.  
            - overcast: The sky is heavely cloudy, no light, not sun or clear sky is visable.
            - other_weather: Any weather situation that doesn't fit the above categories.

        <area_tag> - Define the area type, and the options for the <area_tag> are:
            - highway: The scene takes place on a highway or freeway.  
            - rural: The location is in a rural environment.  
            - terminal: The setting is in a terminal or a logistics hub.  
            - parking: The scene occurs in a parking lot, rest area, or gas station.  
            - city: The setting is in an urban city environment.  
            - residential: The location is in a residential area, typically outside of urban centers.  
            - other_area: Any area that doesn’t fit the other categories.

        <lighting_tag> - Describe the lighting conditions, and the options for the <lighting_tag> are:
            - illuminated: The scene is well-lit.  
            - glare: There is overexposure or very bright light.  
            - dark: The scene is underexposed or dark.  
            - twilight: The light is dim, such as during dusk or dawn.  
            - other_lighting: Any other lighting conditions not covered above.

        <structure_tag> - Define any significant structures, and the options for the <structure_tag> are:
            - tunnel: The scene takes place inside a tunnel.  
            - bridge: The setting is on a bridge, or overpassing another area.  
            - underpass: The scene happens under an overpass, in a underpass.  
            - overpass: The location is on an overpass of roads or rail lines.  
            - regular: No specific structures are involved.

        <construction_tag> - Mention any construction activity, and the options for the <construction_tag> are:
            - roadworks: The scene involves road construction or temporary street layouts.    
            - unchanged: There is no construction activity in the scene.

        [/INST]
        """

    
    # Append the prompt and images to the test data
    model_input = images_base64 + [prompt]

    # Convert the test data to the correct numpy array format
    model_input_array = np.array([model_input])

    # Now pass this input array to the model's predict method
    result = loaded_model.predict(model_input_array)

    cleaned_result = clean_output(result[0])

    # Get the correct daytime and season based on the timestamp
    end_result = add_daytime_and_season(cleaned_result, lst_scenes_samples_timestamps[i])

    print(end_result)

    all_results.append(end_result)

    i += 1
    # Clear CUDA cache only after processing all data
    torch.cuda.empty_cache()
    gc.collect()

```


```python
# Saving the variables in a pickle file so we don't need to run the all the code every time
import pickle

# Define DBFS path for the pickle file
dbfs_path = "/dbfs/FileStore/all_data.pkl"

# Save the variables to a pickle file, including all_selected_images
with open(dbfs_path, "wb") as file:
    pickle.dump({
        "all_results": all_results,
        "lst_scenes_desc": lst_scenes_desc,
        "all_selected_images": all_selected_images,
        "trucksc": trucksc
    }, file)

print(f"Data saved to {dbfs_path}")
```


```python
# Load the variables from the pickle file
import pickle
from truckscenes import TruckScenes

with open("/dbfs/FileStore/all_data.pkl", "rb") as file:
    data = pickle.load(file)
    all_results = data["all_results"]
    lst_scenes_desc = data["lst_scenes_desc"]
    all_selected_images = data["all_selected_images"]
    trucksc = data["trucksc"]

print("Data loaded successfully")
print("all_results:", all_results)
print("lst_scenes_desc:", lst_scenes_desc)
print("all_selected_images:", all_selected_images)
print("trucksc:", trucksc)
```

### Save differences to DBFS


```python
import os
import pickle
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt

# Define DBFS paths
dbfs_output_folder = "/dbfs/FileStore/scene_differences"
differences_txt_path = "/dbfs/FileStore/differences.txt"
detailed_differences_txt_path = "/dbfs/FileStore/differences_detailed.txt"
full_output_pkl_path = "/dbfs/FileStore/full_output.pkl"

# Utility to check and delete files or folders in DBFS if they exist
def ensure_path_does_not_exist(dbfs_path):
    if os.path.exists(dbfs_path):
        if os.path.isdir(dbfs_path):
            # Remove directory recursively
            dbutils.fs.rm(dbfs_path.replace("/dbfs", "dbfs:"), recurse=True)
        else:
            # Remove single file
            dbutils.fs.rm(dbfs_path.replace("/dbfs", "dbfs:"))

# Ensure previous files/folders do not exist by deleting them if present
ensure_path_does_not_exist(dbfs_output_folder)
ensure_path_does_not_exist(differences_txt_path)
ensure_path_does_not_exist(detailed_differences_txt_path)
ensure_path_does_not_exist(full_output_pkl_path)

# Recreate the main output folder in DBFS
os.makedirs(dbfs_output_folder, exist_ok=True)

# Check the lengths of lists to ensure they're the same
if len(all_results) != len(lst_scenes_desc):
    print("Lists must be the same length to compare.")
else:
    differences = []
    detailed_differences = []
    category_counts = Counter()

    # Prepare a list to store each comparison for the .pkl file
    full_output_data = []

    # Process each item in the lists
    for index, (item1, item2) in enumerate(zip(all_results, lst_scenes_desc)):
        # Split the attributes
        attributes1 = item1.split(";")
        attributes2 = item2.split(";")

        # Find the differences in attributes
        item_diffs = []
        title_diff = []
        for attr1, attr2 in zip(attributes1, attributes2):
            if attr1 != attr2:
                category = attr1.split(".")[0]  # Get the category name
                item_diffs.append((attr1, attr2))
                title_diff.append(f"M: {attr1} vs GT: {attr2}")
                category_counts[category] += 1

        # Store item-specific differences if any
        if item_diffs:
            differences.append((index, item1, item2))
            detailed_differences.append((index, item_diffs))

            # Use the provided list of 5 selected images for this scene
            selected_images = all_selected_images[index]

            # Create a title with all category differences
            title_text = " | ".join(title_diff)

            # Generate an image with all differences in the title
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            fig.suptitle(title_text, fontsize=14, weight='bold')

            # Add each selected image to the figure
            for idx, img_path in enumerate(selected_images):
                try:
                    img = Image.open(img_path)
                    axes[idx].imshow(img)
                    axes[idx].axis('off')  # Hide axes for a clean look
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    axes[idx].text(0.5, 0.5, "Error loading image", ha='center', va='center', color="red")

            # Add the scene token as footer text
            scene_token = trucksc.scene[index]['token']
            plt.figtext(0.5, 0.01, f"Scene token: {scene_token}", ha='center', fontsize=12)

            # Save the figure in the DBFS folder
            output_path = os.path.join(dbfs_output_folder, f"scene_{scene_token}.png")
            plt.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved image for scene {scene_token} in {output_path}")

        # Append each pair to full_output_data, regardless of differences
        full_output_data.append({
            f"model_output_{scene_token}": item1,
            f"ground_truth_{scene_token}": item2
        })

    # Write the original differences to a file in DBFS
    with open(differences_txt_path, "w") as file:
        for index, item1, item2 in differences:
            scene_token = trucksc.scene[index]['token']
            file.write(f"Token: {scene_token}\n\"M: {item1}\"\n\"GT: {item2}\"\n\n")

    # Write detailed differences to a separate file in DBFS
    with open(detailed_differences_txt_path, "w") as file:
        for index, item_diffs in detailed_differences:
            scene_token = trucksc.scene[index]['token']
            file.write(f"Token: {scene_token}\n")
            for diff1, diff2 in item_diffs:
                file.write(f"M: {diff1}\nGT: {diff2}\n\n")

        # Write summary of differences by category
        file.write("Summary of differences by category:\n")
        for category, count in category_counts.items():
            file.write(f"{category}: {count}\n")

    # Save the labeled output and ground truth data to a .pkl file in DBFS
    with open(full_output_pkl_path, "wb") as pkl_file:
        pickle.dump(full_output_data, pkl_file)

    print("Differences written to /dbfs/FileStore/differences.txt")
    print("Detailed differences written to /dbfs/FileStore/differences_detailed.txt")
    print("Full output and ground truth saved to /dbfs/FileStore/full_output.pkl")
```
