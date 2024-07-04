from GUI import App
import threading
stop_stream = threading.Condition()

def start_app(app):
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


def stream(app):
    while not app.stop_stream:
        app.video_stream()

NUM_OF_THREADS = 2
functions = [start_app, stream]

def main():
    app = App()
    arguments = [(app,), (app,)]
    threads = []
    for i in range(NUM_OF_THREADS):
        thread = threading.Thread(target=functions[i], args=arguments[i])
        threads.append(thread)
    print("starting threads")
    for thread in threads:
        thread.start()
    print("joining threads")
    for thread in threads:
        thread.join()
    print("threads finished")

if __name__ == "__main__":
    main()