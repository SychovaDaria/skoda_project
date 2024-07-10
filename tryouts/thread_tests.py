from GUI import App
import threading
stop_stream = threading.Condition()

def stream(app):
    global stop_stream
    while True:
        app.video_stream()
        with stop_stream:
            stop_stream.wait(timeout=10)
            break
        
        

NUM_OF_THREADS = 1
functions = [stream]

def main():
    global stop_stream
    app = App()
    arguments = [(app,)]
    threads = []
    for i in range(NUM_OF_THREADS):
        thread = threading.Thread(target=functions[i], args=arguments[i])
        threads.append(thread)
    print("starting threads")
    for thread in threads:
        thread.start()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
    stop_stream = True
    print("joining threads")
    for thread in threads:
        thread.join()
    print("threads finished")

if __name__ == "__main__":
    main()