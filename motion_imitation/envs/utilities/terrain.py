import pybullet as p
import random

def get_uneven_terrain():
    heightPerturbationRange = 0.06
    numHeightfieldRows = 256
    numHeightfieldColumns = 256
    heightfieldData = [0]*numHeightfieldRows*numHeightfieldColumns
    for j in range(int(numHeightfieldColumns/2)):
        for i in range(int(numHeightfieldRows/2)):
            height = random.uniform(0, heightPerturbationRange)
            heightfieldData[2*i+2*j*numHeightfieldRows] = height
            heightfieldData[2*i+1+2*j*numHeightfieldRows] = height
            heightfieldData[2*i+(2*j+1)*numHeightfieldRows] = height
            heightfieldData[2*i+1+(2*j+1)*numHeightfieldRows] = height

    terrainShape = p.createCollisionShape(shapeType=p.GEOM_HEIGHTFIELD, meshScale=[.05, .05, 1], heightfieldTextureScaling=(
    numHeightfieldRows-1)/2, heightfieldData=heightfieldData, numHeightfieldRows=numHeightfieldRows, numHeightfieldColumns=numHeightfieldColumns)
    ground_id = p.createMultiBody(0, terrainShape)

    return ground_id
