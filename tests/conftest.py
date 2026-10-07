"""Router failover state is module-global (it must outlive a request). Clear it
around every test so a cooldown or hold set by one test cannot decide another."""
import pytest

from hello_operator import server as server_mod


@pytest.fixture(autouse=True)
def _clean_failover_state():
    def clear():
        for d in (server_mod._KEY_NEXT, server_mod._KEY_COOLDOWN,
                  server_mod._FREE_EXHAUSTED, server_mod._HOLD):
            d.clear()
    clear()
    yield
    clear()
