"""
AR.I.S — MongoDB Backend Client Service

Provides a clean, decoupled MongoDB connection layer using PyMongo.
Honors Rule 1 (Zero Hard Patches) by providing native connection management
and diagnostic health checks without monkey-patching Django internals.
"""

from django.conf import settings
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

_mongo_client = None


def get_mongo_client():
    """
    Returns the singleton PyMongo client instance.
    Initializes the client if not already connected.
    """
    global _mongo_client
    if _mongo_client is None:
        uri = 'mongodb://localhost:27017/'
        if getattr(settings, 'configured', False):
            uri = getattr(settings, 'MONGODB_URI', uri)
        _mongo_client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _mongo_client


def get_mongo_db():
    """
    Returns the active MongoDB database object configured in settings.
    """
    client = get_mongo_client()
    db_name = 'aris_db'
    if getattr(settings, 'configured', False):
        db_name = getattr(settings, 'MONGODB_DB_NAME', db_name)
    return client[db_name]


def check_mongo_connection():
    """
    Performs an active ping against the MongoDB cluster.
    Returns a dictionary containing connection status and diagnostic information.
    """
    try:
        client = get_mongo_client()
        # The admin command 'ping' verifies server reachability
        client.admin.command('ping')
        db_name = getattr(settings, 'MONGODB_DB_NAME', 'aris_db')
        return {
            'connected': True,
            'database': db_name,
            'message': 'Successfully connected to MongoDB cluster.'
        }
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        return {
            'connected': False,
            'database': getattr(settings, 'MONGODB_DB_NAME', 'aris_db'),
            'message': f'MongoDB connection failed: {str(e)}'
        }
    except Exception as e:
        return {
            'connected': False,
            'database': getattr(settings, 'MONGODB_DB_NAME', 'aris_db'),
            'message': f'Unexpected error checking MongoDB: {str(e)}'
        }
