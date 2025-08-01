import drone

model = drone.model('./runs/detect/train/weights/last.pt')
model.pursueObject(0.6)
