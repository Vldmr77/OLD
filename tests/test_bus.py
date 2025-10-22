import time

from scalp_system.control.bus import BusClient, Event, EventBus


def test_event_bus_dispatches_events():
    bus = EventBus(host="127.0.0.1", port=0)
    received = []
    bus.register("system.test", lambda payload: received.append(payload.get("value")))
    bus.start()
    try:
        client = BusClient(host="127.0.0.1", port=bus.port)
        assert client.check_available()
        client.emit_event(Event("system.test", {"value": 42}))
        timeout = time.time() + 1.0
        while not received and time.time() < timeout:
            time.sleep(0.01)
    finally:
        bus.stop()
    assert received == [42]


def test_bus_client_reports_unavailable_port():
    bus = EventBus(host="127.0.0.1", port=0)
    bus.start()
    port = bus.port
    bus.stop()
    client = BusClient(host="127.0.0.1", port=port)
    assert client.check_available() is False
