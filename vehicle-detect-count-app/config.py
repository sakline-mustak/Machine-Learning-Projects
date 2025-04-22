class Config(object):
    SECRET_KEY = 'vehicle_detect'
    SESSION_COOKIE_SECURE = True
    DEFAULT_THEME = None

class DevelopmentConfig(Config):
    DEBUG = True