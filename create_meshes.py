#quick usage note: download and install huggingface diffusers using anaconda, a python tool, like this:
#conda install -c conda-forge diffusers
#then create one or more .csv files in the same directory as the script with the following columns. The only columns used are 0, 1, 2, and 8 so the rest can be empty. 
#example:
#category,name,prompt,negative,my prompt,my negative,my name,my number,my seed mod,calculated seed, skip
#see https://docs.google.com/spreadsheets/d/17E4NR7sOZUWKSrFGfJULz7DzSEbZE65gMczFbDKmmnI/edit?usp=sharing

from diffusers import DiffusionPipeline, UniPCMultistepScheduler
from compel import Compel
import torch
import csv
import os
import os.path
from subprocess import call
import argparse
import logging
import time

import numpy as np
import rembg
import xatlas
from PIL import Image

import sys

#add TripoSR to the syspath temporarily so we can import and run tsr code
sys.path.insert(0, os.path.abspath('TripoSR'))

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, save_video
from tsr.bake_texture import bake_texture

#pipe = DiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16)
#pipe.to("cuda")

triposr_device = "cuda:0"
triposr_model="stabilityai/TripoSR"
tripsor_chunksize = 8192
triposr_foreground_ratio = .85
triposr_bake_texture = False
triposr_texture_resolution = 2048
triposr_marching_cubes_resolution = 256

if not torch.cuda.is_available():
    triposr_device = "cpu"

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"

    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
        logging.info(f"{name} ...")

    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        logging.info(f"{name} finished in {t:.2f}{self.time_unit}.")

timer = Timer()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)

timer.start("Initializing model")

model = TSR.from_pretrained(
    triposr_model,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(tripsor_chunksize)
model.to(triposr_device)
timer.end("Initializing model")

#loop through all .csv files in the current directory, and generate images from the contents 
for x in os.listdir():
    if x.endswith(".csv"):

        print("loading " + x)

        with open(x) as csv_file:
            #open the csv reader
            csv_reader = csv.reader(csv_file, delimiter=',')
            count = 0
            for row in csv_reader:
                #for each row, but skip the first row that contains names
                count = count + 1
                if count == 1:
                    continue

                #get values, pass as prompts, name the file, etc. 
                category = row[0]
                name = row[1]
                prompt = row[2]
                negative = row[3]
                seed = int(row[12])
                skip = row[13]

                if skip == "y":
                    continue

                #create an output directory for each category if needed
                directory_path = "output/" + category + "/"

                if not os.path.isdir(directory_path): 
                    os.makedirs(directory_path) 

                image_path = directory_path + name + ".png"
                out_mesh_path = directory_path + name + ".obj"
                
                #check if the image already exists, if not, generate it using huggingface diffusers
                # if os.path.isfile(image_path) == False:

                #     print("generating " + image_path)
                
                #     #prompt_embeds = compel_proc(prompt)
                #     #negative_embeds = compel_proc(negative)

                #     generator = torch.Generator(device="cuda").manual_seed(seed)
                #     #image = pipe(prompt_embeds=prompt_embeds, generator=generator, negative_prompt_embeds=negative_embeds, num_inference_steps=20).images[0]
                #     image = pipe(prompt, negative_prompt=negative).images[0]
                    
                #     print("saving " + image_path)

                #     image.save(image_path)

                # else:
                #     print("skipping " + image_path)

                if os.path.isfile(out_mesh_path) == False and os.path.isfile(image_path) == True:

                    print("generating " + out_mesh_path)

                    #also run triposr
                    #call("python TripoSR/run.py " + directoryPath + name + ".png" + " --output-dir " + directoryPath + name , shell=True)
                    rembg_session = rembg.new_session()

                    triposr_image = remove_background(Image.open(image_path), rembg_session)
                    triposr_image = resize_foreground(triposr_image, triposr_foreground_ratio)
                    triposr_image = np.array(triposr_image).astype(np.float32) / 255.0
                    triposr_image = triposr_image[:, :, :3] * triposr_image[:, :, 3:4] + (1 - triposr_image[:, :, 3:4]) * 0.5
                    triposr_image = Image.fromarray((triposr_image * 255.0).astype(np.uint8))
                    #if not os.path.exists(os.path.join(output_dir, str(i))):
                    #    os.makedirs(os.path.join(output_dir, str(i)))
                    triposr_image.save(directory_path + name + "_bgremoved.png")

                    logging.info(f"Running image " + image_path + "...")

                    timer.start("Running model")
                    with torch.no_grad():
                        scene_codes = model([triposr_image], device=triposr_device)
                    timer.end("Running model")

                    timer.start("Extracting mesh")
                    meshes = model.extract_mesh(scene_codes, not triposr_bake_texture, resolution=triposr_marching_cubes_resolution)
                    timer.end("Extracting mesh")

                    if triposr_bake_texture:
                        out_texture_path = directory_path + name + "_baked.png"

                        timer.start("Baking texture")
                        bake_output = bake_texture(meshes[0], model, scene_codes[0], triposr_texture_resolution)
                        timer.end("Baking texture")

                        timer.start("Exporting mesh and texture")
                        xatlas.export(out_mesh_path, meshes[0].vertices[bake_output["vmapping"]], bake_output["indices"], bake_output["uvs"], meshes[0].vertex_normals[bake_output["vmapping"]])
                        #note this probably why our texture was flipped - the FLIP_TOP_BOTTOM
                        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(Image.FLIP_TOP_BOTTOM).save(out_texture_path)
                        timer.end("Exporting mesh and texture")
                    else:
                        timer.start("Exporting mesh")
                        meshes[0].export(out_mesh_path)
                        timer.end("Exporting mesh")
                else: 
                    print("skipping " + out_mesh_path)



