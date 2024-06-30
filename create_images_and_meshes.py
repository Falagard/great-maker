from subprocess import call

call("python create_images.py", shell=True)
call("python create_meshes.py", shell=True)