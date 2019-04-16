import os
import time
import hashlib
import json
import paho.mqtt.client as mqtt
import logging
import jsonschema

import numpy as np
SERVICE_NAME = "tf-pose"


def rpc(client, userdata, message):
    try:
        request = json.loads(message.payload.decode('utf-8'))
        reply_to = request.get('replyTo', 'error')
        method = request['method']
        params = request['params']

        if method == 'start':
            response = start_stream(params, client)
        elif method == 'stop':
            pass
        else:
            raise ValueError("Unknown method '{}'".format(method))
        response['apiVersion'] = 'beta'
        response['context'] = request['context']
        response['method'] = method
        response['serviceName'] = SERVICE_NAME
    except jsonschema.ValidationError as e:
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
        response = {'data': {'streamId': 'stream_id'}}
    except Exception as e:
        print(e)
        raise
    return response


def on_connect(client, userdata, flags, rc):
    route_in_keys = userdata['route_in_keys']
    print('Client connected -- Listening to routes: {}'.format(route_in_keys))
    client.subscribe(list(zip(route_in_keys, [1] * len(route_in_keys))))
    client.publish(userdata['health'], 'healthy', qos=1, retain=True)


def create_client():
    try:
        client = mqtt.Client()
        client.on_connect = on_connect
        # userdata = {
        #     'route_in_keys': ['$share/{}/{}'.format(SERVICE_NAME, sub) for subkey, sub in conf[SERVICE_NAME].items() if subkey.endswith("sub") or subkey == 'rpc'],
        #     'health': '/'.join([conf['all']['health_stub'], os.environ['HOSTNAME']]),
        # }
        # client.user_data_set(userdata)
        #client.will_set(userdata['health'], 'unhealthy', qos=1, retain=True)
        client.username_pw_set(username='username',
                               password='password')
        client.message_callback_add(SERVICE_NAME, rpc)
        client.connect_async(host='host')
    except Exception as e:
        raise Exception("Failed to create client with error {}".format(e))
    return client


if __name__ == '__main__':
    try:
        client = create_client()
        client.loop_forever()
    except Exception as e:
        print(e)
        raise Exception(e)
