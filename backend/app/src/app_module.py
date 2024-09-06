# src/app_module.py

from nest.core import Module, PyNestFactory
from .app_controller import AppController
from .app_service import AppService


@Module(
    controllers=[AppController],
    providers=[AppService],
)
class AppModule:
    pass


app = PyNestFactory.create(
    AppModule,
    description="This is my PyNest app",
    title="My App",
    version="1.0.0",
    debug=True,
)

http_server = app.get_server()
