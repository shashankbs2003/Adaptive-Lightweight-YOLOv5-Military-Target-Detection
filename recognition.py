from sklearn.metrics.pairwise import pairwise_distances
from tensorflow.python.platform import gfile
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import detect_and_align
import argparse
import easygui
import time
import cv2
import os
import time
import sys
import time
import random
import datetime
import random
from tkinter import *
from PIL import Image,ImageTk

rec=1      
class IdData:
    """Keeps track of known identities and calculates id matches"""

    def __init__(
        self, id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, distance_treshold
    ):
        print("Loading known identities: ", end="")
        self.distance_treshold = distance_treshold
        self.id_folder = id_folder
        self.mtcnn = mtcnn
        self.id_names = []
        self.embeddings = None

        image_paths = []
        os.makedirs(id_folder, exist_ok=True)
        ids = os.listdir(os.path.expanduser(id_folder))
        if not ids:
            return

        for id_name in ids:
            id_dir = os.path.join(id_folder, id_name)
            image_paths = image_paths + [os.path.join(id_dir, img) for img in os.listdir(id_dir)]

        print("Found %d images in id folder" % len(image_paths))
        aligned_images, id_image_paths = self.detect_id_faces(image_paths)
        feed_dict = {images_placeholder: aligned_images, phase_train_placeholder: False}
        self.embeddings = sess.run(embeddings, feed_dict=feed_dict)

        if len(id_image_paths) < 5:
            self.print_distance_table(id_image_paths)

    def add_id(self, embedding, new_id, face_patch):
        if self.embeddings is None:
            self.embeddings = np.atleast_2d(embedding)
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        self.id_names.append(new_id)
        id_folder = os.path.join(self.id_folder, new_id)
        os.makedirs(id_folder, exist_ok=True)
        filenames = [s.split(".")[0] for s in os.listdir(id_folder)]
        numbered_filenames = [int(f) for f in filenames if f.isdigit()]
        img_number = max(numbered_filenames) + 1 if numbered_filenames else 0
        cv2.imwrite(os.path.join(id_folder, f"{img_number}.jpg"), face_patch)

    def detect_id_faces(self, image_paths):
        aligned_images = []
        id_image_paths = []
        for image_path in image_paths:
            image = cv2.imread(os.path.expanduser(image_path), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_patches, _, _ = detect_and_align.detect_faces(image, self.mtcnn)
            if len(face_patches) > 1:
                print(
                    "Warning: Found multiple faces in id image: %s" % image_path
                    + "\nMake sure to only have one face in the id images. "
                    + "If that's the case then it's a false positive detection and"
                    + " you can solve it by increasing the thresolds of the cascade network"
                )
            aligned_images = aligned_images + face_patches
            id_image_paths += [image_path] * len(face_patches)
            path = os.path.dirname(image_path)
            self.id_names += [os.path.basename(path)] * len(face_patches)

        return np.stack(aligned_images), id_image_paths

    def print_distance_table(self, id_image_paths):
        """Prints distances between id embeddings"""
        distance_matrix = pairwise_distances(self.embeddings, self.embeddings)
        image_names = [path.split("/")[-1] for path in id_image_paths]
        print("Distance matrix:\n{:20}".format(""), end="")
        [print("{:20}".format(name), end="") for name in image_names]
        for path, distance_row in zip(image_names, distance_matrix):
            print("\n{:20}".format(path), end="")
            for distance in distance_row:
                print("{:20}".format("%0.3f" % distance), end="")
        print()

    def find_matching_ids(self, embs):
        if self.id_names:
            matching_ids = []
            matching_distances = []
            distance_matrix = pairwise_distances(embs, self.embeddings)
            for distance_row in distance_matrix:
                min_index = np.argmin(distance_row)
                if distance_row[min_index] < self.distance_treshold:
                    matching_ids.append(self.id_names[min_index])
                    matching_distances.append(distance_row[min_index])
                else:
                    matching_ids.append(None)
                    matching_distances.append(None)
        else:
            matching_ids = [None] * len(embs)
            matching_distances = [np.inf] * len(embs)
        return matching_ids, matching_distances

def load_model(model):
    model_exp = os.path.expanduser(model)
    if os.path.isfile(model_exp):
        print("Loading model filename: %s" % model_exp)
        with gfile.FastGFile(model_exp, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    else:
        raise ValueError("Specify model file, not directory!")

scanned =0
amount=1000
user =0

with tf.Graph().as_default():
    with tf.Session() as sess:

        # Setup models
        mtcnn = detect_and_align.create_mtcnn(sess, None)

        model= '20170512-110547/20170512-110547.pb'
        load_model(model)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        id_folder = '20170512-110547/ids'
        # Load anchor IDs
        id_data = IdData(
            id_folder, mtcnn, sess, embeddings, images_placeholder, phase_train_placeholder, 1.0
        )

        cap = cv2.VideoCapture(0)
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        show_landmarks = False
        show_bb = False
        show_id = True
        show_fps = False
        frame_detections = None
        Name = ''
        scanned_next =0
        global count
        count=0
        kcounter = 0
        ucounter = 0
        tcounter = 0
        while True:

            start = time.time()


            _, frame = cap.read()

            # Locate faces and landmarks in frame
            face_patches, padded_bounding_boxes, landmarks = detect_and_align.detect_faces(frame, mtcnn)

            if len(face_patches) > 0:
                tcounter += 1
                face_patches = np.stack(face_patches)
                feed_dict = {images_placeholder: face_patches, phase_train_placeholder: False}
                embs = sess.run(embeddings, feed_dict=feed_dict)

                matching_ids, matching_distances = id_data.find_matching_ids(embs)
                frame_detections = {"embs": embs, "bbs": padded_bounding_boxes, "frame": frame.copy()}
                
                print("Matches in frame:")
                for bb, landmark, matching_id, dist in zip(
                    padded_bounding_boxes, landmarks, matching_ids, matching_distances
                ):
                    
                    if matching_id is None:
                        matching_id = "Intruder"
                        ucounter += 1
                        
                        
                        print("Intruder! Couldn't find match.")                                
                        count=0
                    else:
                        print("Hi %s! Distance: %1.4f" % (matching_id, dist))
                        kcounter += 1
                        with open("user_details.txt", "r") as file1:
                            data = file1.readlines()
                            print(data)
                            if matching_id in str(data):
                                print("person identified")
                                

                    if show_id:
                        Name = matching_id
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, matching_id, (bb[0], bb[3]), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                print("Couldn't find a face")

            cv2.imshow("frame", frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        if ucounter > kcounter:
            print('unknown person detected')
        else:
            print(Name+' detected')
