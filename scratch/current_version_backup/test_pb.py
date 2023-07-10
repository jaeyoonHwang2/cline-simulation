import ns3gym.messages_pb2 as pb

dataContainer = pb.DataContainer()
dataContainer.type = pb.Box

boxContainerPb = pb.BoxDataContainer()
actions = (1564878020, 492660)
shape = [len(actions)]
boxContainerPb.shape.extend(shape)
boxContainerPb.dtype = pb.UINT
boxContainerPb.uintData.extend(actions)
dataContainer.data.Pack(boxContainerPb)

print("boxContainerPb: ", boxContainerPb)
print("dataContainer: ", dataContainer)

dataContainerUnpack = pb.DataContainer()
# dataContainerUnpack.type = pb.Box
result = dataContainerUnpack.data.Unpack(boxContainerPb)
boxContainerPbUnpack = pb.BoxDataContainer()
dataContainerUnpack.data.Pack(boxContainerPbUnpack)
print("dataContainerUnpack (unpacked)", dataContainerUnpack.data)