from paho.mqtt import client as mc
import time

broker = 'mqtt.eclipseprojects.io'
client_id ='id1'


def my_call(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print("Failed to connect, return code %d\n", rc)

def on_message(client, userdata, msg):
    print("hi")
    print("Received",str(msg.payload))

client = mc.Client('hi')
client.on_connect = my_call
client.connect(broker)

client.loop_start()
sub=client.subscribe('int')
result=sub[0]
if(result==0):
    print("Subscribtion successful")
else:
    print("Subscribtion not successful")
sub=client.subscribe('char')
result=sub[0]
if(result==0):
    print("Subscribtion successful")
else:
    print("Subscribtion not successful")
sub=client.subscribe('str')
result=sub[0]
if(result==0):
    print("Subscribtion successful")
else:
    print("Subscribtion not successful")

client.on_message = on_message
time.sleep(30)
client.loop_stop()

