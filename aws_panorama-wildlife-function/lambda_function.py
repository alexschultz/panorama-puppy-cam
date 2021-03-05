import panoramasdk
import cv2
import numpy as np
import boto3
from Processor import Processor
# use urllib instead of requests because it's part of the standard lib which simplifies the lambda deployment process
from urllib.request import urlopen
import socket

# set default socket timeout in case the alarm server doesn't respond
timeout = 1
socket.setdefaulttimeout(timeout)


HEIGHT = 640
WIDTH = 640
class_list = ["corgi", "cayote", "deer"]


class wildlife_counter(panoramasdk.base):

    def interface(self):
        return {
            "parameters":
                (
                    ("float", "threshold", "Minimum confidence for display", 0.50),
                    ("model", "wildlife", "Name of the model in AWS Panorama",
                     "panorama-wildlife-v2"),
                    ("int", "batch_size", "Model batch size", 1)
                ),
            "inputs":
                (
                    ("media[]", "video_in", "Camera input stream"),
                ),
            "outputs":
                (
                    ("media[video_in]", "video_out", "Camera output stream"),
                )
        }

    def init(self, parameters, inputs, outputs):
        print("init")
        try:
            self.consecutive_cayote_frames = 0
            self.consecutive_cayote_threshold = 5
            self.warning_server_url = "http://garage.home.com/trigger"
            self.processor = Processor(len(class_list))
            self.threshold = parameters.threshold
            self.frame_num = 0
            self.colours = np.random.rand(32, 3)

            print("Loading model: " + parameters.wildlife)
            self.model = panoramasdk.model()
            self.model.open(parameters.wildlife, 1)

            pred_info = self.model.get_output(0)
            prob_info = self.model.get_output(1)
            rect_info = self.model.get_output(2)

            self.pred_array = np.empty(
                pred_info.get_dims(), dtype=pred_info.get_type())

            self.prob_array = np.empty(
                prob_info.get_dims(), dtype=prob_info.get_type())

            self.rect_array = np.empty(
                rect_info.get_dims(), dtype=rect_info.get_type())

            print("Initialization complete")
            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    def trigger_warning(self):
        with urlopen(self.warning_server_url) as response:
            response_content = response.read()
            print(response_content)

    def preprocess(self, img):
        resized = cv2.resize(img, (HEIGHT, WIDTH))
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]

        img = resized.astype(np.float32) / 255.
        img_a = img[:, :, 0]
        img_b = img[:, :, 1]
        img_c = img[:, :, 2]

        # Normalize data in each channel
        # img_a = (img_a - mean[0]) / std[0]
        # img_b = (img_b - mean[1]) / std[1]
        # img_c = (img_c - mean[2]) / std[2]

        # Put the channels back together
        x1 = [[[], [], []]]
        x1[0][0] = img_a
        x1[0][1] = img_b
        x1[0][2] = img_c

        x1 = np.asarray(x1)
        return x1

    def entry(self, inputs, outputs):
        self.frame_num += 1

        for i in range(len(inputs.video_in)):
            stream = inputs.video_in[i]
            wildlife_image = stream.image

            # Prepare the image and run inference
            x1 = self.preprocess(wildlife_image)

            # get the original image shape
            # shape_orig_wh = wildlife_image.shape[:2]

            self.model.batch(0, x1)
            self.model.flush()
            resultBatchSet = self.model.get_result()

            pred_batch = resultBatchSet.get(0)
            prob_batch = resultBatchSet.get(1)
            rect_batch = resultBatchSet.get(2)

            pred_batch.get(0, self.pred_array)
            prob_batch.get(0, self.prob_array)
            rect_batch.get(0, self.rect_array)

            output = [self.pred_array, self.prob_array, self.rect_array]

            boxes, confs, classes = self.processor.post_process(
                output, conf_thres=self.threshold)

            cayote_spotted = False

            for box, conf, cls in zip(boxes, confs, classes):
                if cls == 1:
                    self.consecutive_cayote_frames = self.consecutive_cayote_frames + 1
                    cayote_spotted = True

                # draw rectangle
                x1, y1, x2, y2 = box
                print(f"({x1},{y1}):({x2},{y2})")
                cf = conf[0]
                left = np.clip(x1 / np.float(HEIGHT), 0, 1)
                top = np.clip(y1 / np.float(WIDTH), 0, 1)
                right = np.clip(x2 / np.float(HEIGHT), 0, 1)
                bottom = np.clip(y2 / np.float(WIDTH), 0, 1)
                stream.add_rect(left, top, right, bottom)
                cls_name = class_list[cls]
                stream.add_label(f"{cls_name}: {str(cf)}", right, bottom)
                print(f"{cls_name}:{str(cf)}")

            if cayote_spotted:
                print(
                    f"cayote spotted count: {str(self.consecutive_cayote_frames)}")
                if self.consecutive_cayote_frames >= self.consecutive_cayote_threshold:
                    try:
                        self.trigger_warning()
                        # reset the counter so we don't spam the warning service
                        self.consecutive_cayote_frames = 0
                    except Exception as e:
                        print("Exception: {}".format(e))
            else:
                # we didn't see any cayotes in this frame, reset the counter
                self.consecutive_cayote_frames = 0

            self.model.release_result(resultBatchSet)
            outputs.video_out[i] = stream

        return True


def main():
    print("main")
    wildlife_counter().run()


main()
