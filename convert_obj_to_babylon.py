import bpy
bpy.ops.wm.obj_import(filepath='D:\\Downloads\\TripoSR-main\\TripoSR-main\\output\\0\\mesh.obj')
bpy.context.object.rotation_euler[0] = 0
material = bpy.data.materials['Material']
material.use_nodes = True
bsdf = material.node_tree.nodes["Principled BSDF"]
texImage = material.node_tree.nodes.new('ShaderNodeTexImage')
texImage.image = bpy.data.images.load('d:\\Downloads\\TripoSR-main\\TripoSR-main\\output\\0\\texture.png')
material.node_tree.links.new(bsdf.inputs['Base Color'], texImage.outputs['Color'])
obj = bpy.context.selected_objects[0]
obj.data.materials.append(material)
bpy.ops.object.modifier_add(type='WELD')
bpy.ops.object.modifier_add(type='DECIMATE')
bpy.context.object.modifiers["Decimate"].ratio = 0.025
bpy.ops.object.modifier_add(type='WEIGHTED_NORMAL')
bpy.ops.export.bjs(filepath="D:\\src\\AlexCentauri\\centauri-game-engine\\sample-assets\\export.babylon")