import time
import utils
import panoramasdk
import cv2
import numpy as np
import boto3


INPUT_SIZE = 640
MODEL = 'yolov5s-wildlife'
THRESHOLD = 0.5
CLASSES = ["corgi", "coyote", "deer"]


class YOLOv5(panoramasdk.base):

    def interface(self):
        return {
            "parameters":
                (
                    ("float", "threshold", "Detection threshold", THRESHOLD),
                    ("model", "model", "YOLOv5 wildlife model", MODEL),
                    ("int", "input_size",
                     "Model input size (actual shape will be [1, 3, input_size, input_size])", INPUT_SIZE)
                ),
            "inputs": (("media[]", "video_in", "Camera input stream"),),
            "outputs": (("media[video_in]", "video_out", "Camera output stream"),)
        }

    def log(self, msg, frequency=1000):
        """Log the messages at a reduced rate, can reduce AWS CloudWatch logs polluting and simplify log searching
        params:
            msg (str): message to log
            frequency (int): message will be logged only once for this number of processed image frames
        """
        if self.frame_num % frequency == 1:
            print(f'[Frame: {self.frame_num}]: {msg}')

    def run_inference(self, image):
        self.log("Running inference")
        start = time.time()
        self.model.batch(0, self.processor.preprocess(image))
        self.model.flush()
        result = self.model.get_result()
        inf_time = time.time() - start
        self.log(
            f'Inference completed in {int(inf_time * 1000):,} msec', frequency=1000)

        batch_0 = result.get(0)
        batch_1 = result.get(1)
        batch_2 = result.get(2)
        batch_0.get(0, self.pred_0)
        batch_1.get(0, self.pred_1)
        batch_2.get(0, self.pred_2)
        self.model.release_result(result)

        return inf_time

    def init(self, parameters, inputs, outputs):
        print('init()')
        try:
            # NOTE: these are specific to the alarm server
            self.consecutive_coyote_frames = 0
            self.consecutive_coyote_threshold = 5
            self.warning_server_url = "http://garage.home.com/trigger"

            self.input_size = parameters.input_size
            self.class_names = CLASSES
            self.processor = utils.Processor(
                self.class_names, self.input_size, keep_ratio=True)
            self.threshold = parameters.threshold
            self.frame_num = 0

            # Load model from the specified directory.
            print(f'Loading model {parameters.model}')
            start = time.time()
            self.model = panoramasdk.model()
            self.model.open(parameters.model, 1)
            print(f'Model loaded in {int(time.time() - start)} seconds')

            # Create output arrays
            info_0 = self.model.get_output(0)
            info_1 = self.model.get_output(1)
            info_2 = self.model.get_output(2)

            self.pred_0 = np.empty(info_0.get_dims(), dtype=info_0.get_type())
            self.pred_1 = np.empty(info_1.get_dims(), dtype=info_1.get_type())
            self.pred_2 = np.empty(info_2.get_dims(), dtype=info_2.get_type())

            # Use all the model outputs
            self.predictions = [self.pred_0, self.pred_1, self.pred_2]

            return True

        except Exception as e:
            print("Exception: {}".format(e))
            return False

    def entry(self, inputs, outputs):
        try:
            self.frame_num += 1
            coyote_spotted = False
            for i in range(len(inputs.video_in)):
                stream = inputs.video_in[i]
                in_image = stream.image

                inf_time = self.run_inference(in_image)

                start = time.time()
                (boxes, scores, cids), _ = self.processor.post_process(
                    self.predictions, in_image.shape, self.threshold)
                post_time = time.time() - start

                for (x1, y1, x2, y2), score, cid in zip(boxes, scores, cids):
                    label = f'{self.class_names[int(cid)]} {score:.2f}'
                    stream.add_rect(x1, y1, x2, y2)
                    # (x, y) here are coords of top-left label location
                    stream.add_label(label, x1, y1)

                    # did we detect a coyote
                    if not coyote_spotted and int(cid) == 1:
                        self.consecutive_coyote_frames = self.consecutive_coyote_frames + 1
                        coyote_spotted = True

            if coyote_spotted:
                print(
                    f"coyote spotted count: {str(self.consecutive_coyote_frames)}")
                if self.consecutive_coyote_frames >= self.consecutive_coyote_threshold:
                    try:
                        utils.trigger_warning(
                            remote_server_url=self.warning_server_url)
                        # reset the counter so we don't spam the warning service
                        self.consecutive_coyote_frames = 0
                    except Exception as e:
                        print("Exception: {}".format(e))
            else:
                # we didn't see any coyotes in this frame, reset the counter
                self.consecutive_coyote_frames = 0

            # Visual log of frames and processing times, remove if not needed
            msg = f'Frame: {self.frame_num:,}\n- infer: {int(inf_time * 1000)} msec\n- post-proc: {int(post_time * 1000)} msec'
            stream.add_label(msg, 0.6, 0.1)

            outputs.video_out[i] = stream

        except Exception as e:
            print("Exception: {}".format(e))
            return False

        return True


def main():
    YOLOv5().run()


main()
