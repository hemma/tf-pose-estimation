import base64
import json
import time
import uuid
from threading import Thread

import cv2
import paho.mqtt.client as mqtt
import redis
import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

SERVICE_NAME = "tf-pose"
EXPIRATION_TIME = 180
stream_running = False


def rpc(client, userdata, message):
    try:
        # print('Got message %s' % (message.payload.decode('utf-8'),))
        request = json.loads(message.payload.decode('utf-8'))
        reply_to = request.get('replyTo', 'error')
        method = request['method']
        params = request['params']

        if method == 'start':
            response = start_stream(params, client)
        elif method == 'stop':
            response = stop_stream()
        else:
            raise ValueError("Unknown method '{}'".format(method))
        response['apiVersion'] = 'beta'
        # response['context'] = request['context']
        # response['method'] = method
        response['serviceName'] = SERVICE_NAME
    except Exception as e:
        print(e)
        error_msg = _generate_error_message(e)
        response = {'context': request.get('context', ''),
                    'error': {'code': 1,
                              'message': error_msg,
                              'serviceName': SERVICE_NAME}}
    except Exception as e:
        print(e)
        response = {'error': {'code': 1,
                              'message': str(e),
                              'serviceName': SERVICE_NAME}}
        reply_to = 'error'
    client.publish(reply_to, json.dumps(response))


def _generate_error_message(error):
    error_type = error.validator
    if error_type == 'required':
        defined_params = error.instance
        missing = [k for k in error.validator_value if k not in defined_params]
        msg = 'Missing required param(s): \'{}\''.format(", ".join(missing))
    elif error_type == 'type':
        msg = 'Incompatible type for param: \'{}\''.format("/".join(error.absolute_path))
    elif error_type == 'enum':
        msg = 'Expected param \'{}\' to be one of: {}'.format("/".join(error.absolute_path),
                                                              ", ".join(error.validator_value))
    else:
        print('Unknown cause, type: {}'.format(error_type))
        msg = 'Unknown error.'
    return msg


def start_stream(params, client):
    try:
        global stream_running
        stream_running = True
        Thread(name='publisher', target=on_stream, args=(client, params)).start()

        response = {'data': {'streamId': 'stream_id'}}
    except Exception as e:
        print(e)
        raise
    return response


def stop_stream():
    global stream_running
    stream_running = False
    return {'status': 'stopped'}


def on_stream(client, params):
    redis_store = redis.Redis(host='redis', password='password', port=6379, db=0)
    global stream_running
    print(params)
    gpu_options = tf.GPUOptions()
    gpu_options.allow_growth = True

    w, h = model_wh(params['resize'])
    if w > 0 and h > 0:
        e = TfPoseEstimator(get_graph_path(params['model']), target_size=(w, h),
                            tf_config=tf.ConfigProto(gpu_options=gpu_options))
    else:
        e = TfPoseEstimator(get_graph_path(params['model']), target_size=(432, 368),
                            tf_config=tf.ConfigProto(gpu_options=gpu_options))
    cam = cv2.VideoCapture(params['camera'])
    ret_val, image = cam.read()

    while stream_running:
        ret_val, image = cam.read()
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=params['resize_out_ratio'])

        all_humans = []
        for human in humans:
            body_parts = []
            for part in human.body_parts.values():
                body_parts.append({'index': part.part_idx, 'x': part.x, 'y': part.y, 'score': part.score})
            all_humans.append(body_parts)

        image_id = str(uuid.uuid4())

        retval, buffer = cv2.imencode('.png', image)
        png = base64.b64encode(buffer)

        redis_store.set(image_id, png, ex=EXPIRATION_TIME)

        frame = {'humans': all_humans, 'image': image_id}
        frame_json = json.dumps(frame)

        client.publish(SERVICE_NAME + '/pose', frame_json)


def on_connect(client, userdata, flags, rc):
    print('Client connected -- Listening to %s' % (SERVICE_NAME,))
    client.subscribe(SERVICE_NAME)


def create_client():
    try:
        c = mqtt.Client()
        c.on_connect = on_connect
        c.on_message = rpc
        c.connect(host='mqtt')
    except Exception as e:
        raise Exception("Failed to create client with error {}".format(e))

    print("connected")
    return c


if __name__ == '__main__':
    try:
        time.sleep(3)
        client = create_client()
        client.loop_forever()
    except Exception as e:
        print(e)
        raise Exception(e)
