import zmq
import zmq.asyncio
from tornado.ioloop import IOLoop
from functools import partial

class Sender:
    """
    Create one instance of this in the producer script, to send data to
    a bokeh server.
    """
    def __init__(self, port):
        context = zmq.Context.instance()
        self.socket = context.socket(zmq.PUB)
        self.socket.bind(f'tcp://127.0.0.1:{port}')

    def send(self, step, key, data):
        """
        Sends data to the server.  
        The receiver maintains nested map of step => (key => data).
        sending a (step, key) pair more than once overwrites the existing data.
        """
        self.socket.send_pyobj((step, key, data))

class Receiver:
    def __init__(self, port, doc, callback, *args):
        """
        port: the localhost port to listen on
        doc: the bokeh Document object
        callback: function called as callback(self.data, step, *self.args)
                  each time there is new data received
        self.data is a map of step => (key => custom data)
        """
        context = zmq.asyncio.Context.instance()
        self.socket = context.socket(zmq.SUB)
        self.socket.connect(f'tcp://127.0.0.1:{port}')
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.doc = doc
        self.data = {}
        self.callback = callback
        self.args = args

    async def __call__(self):
        while True:
            step, key, data = await self.socket.recv_pyobj()
            m = self.data.setdefault(step, {})
            m[key] = data
            wrapped = partial(self.callback, self.data, step, *self.args)
            self.doc.add_next_tick_callback(wrapped)
            # print(f'received step {step}')

    def start(self):
        """
        Call this in the bokeh server code at the end of the script.
        This starts the receiver listening for data updates from the
        sender.
        """
        IOLoop.current().spawn_callback(self)


