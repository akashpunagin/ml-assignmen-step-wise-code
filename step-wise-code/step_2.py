import sys
import dlib
from skimage import io

# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "models/shape_predictor_68_face_landmarks.dat"

file_name = "people/bill_gates.jpg"

face_detector = dlib.get_frontal_face_detector()
face_pose_predictor = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

image = io.imread(file_name)

detected_faces = face_detector(image, 1)

print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

win.set_image(image)

for i, face_rect in enumerate(detected_faces):

	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}"
	.format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())
	)

	win.add_overlay(face_rect)

	# get post of the face
	pose_landmarks = face_pose_predictor(image, face_rect)

	win.add_overlay(pose_landmarks)

# wait for user to hit enter	        
dlib.hit_enter_to_continue()