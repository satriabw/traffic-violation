import pytest
from shared.db.connection import get_connection
from shared.db.init import init_db


@pytest.fixture
def con():
    connection = get_connection(":memory:")
    init_db(connection)
    yield connection
    connection.close()
