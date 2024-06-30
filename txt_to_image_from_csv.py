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
from subprocess import call


pipe = DiffusionPipeline.from_pretrained("dreamlike-art/dreamlike-photoreal-2.0", torch_dtype=torch.float16)
#pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

#compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

#loop through all .csv files in the current directory, and generate images from the contents 

for x in os.listdir():
    if x.endswith(".csv"):
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
                directoryPath = "output/" + category + "/"

                if not os.path.isdir(directoryPath): 
                    os.makedirs(directoryPath) 

                #prompt_embeds = compel_proc(prompt)
                #negative_embeds = compel_proc(negative)

                generator = torch.Generator(device="cuda").manual_seed(seed)
                #image = pipe(prompt_embeds=prompt_embeds, generator=generator, negative_prompt_embeds=negative_embeds, num_inference_steps=20).images[0]
                image = pipe(prompt, negative_prompt=negative).images[0]
                image.save(directoryPath + name + ".png")

                #also run triposr
                #call("python TripoSR/run.py " + directoryPath + name + ".png" + " --output-dir " + directoryPath + name , shell=True)



