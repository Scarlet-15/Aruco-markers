import time
from paho.mqtt import client as mc

broker = 'mqtt.eclipseprojects.io'
client_id='id1'

def my_call(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)


def publish(client):
    time.sleep(1)
    msg = 30
    result = client.publish('int', msg)
    status = result[0]
    if status == 0:
        print(msg,"in topic int")
    else:
        print("Message not sent {msg}")
    msg = 'a'
    result = client.publish('char', msg)
    status = result[0]
    if status == 0:
        print(msg)
    else:
        print("Message not sent {msg}")
    msg = "Hi this is Mhanjhu"
    result = client.publish('str', msg)
    status = result[0]
    if status == 0:
        print(msg)
    else:
        print("Message not sent {msg}")
        

client = mc.Client('id1')
client.on_connect = my_call
client.connect(broker)

client.loop_start()
publish(client)
client.loop_stop()

