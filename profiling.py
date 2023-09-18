from testing.testCollection import full_init, start_test
from configs import app

ports = {"social": 30628, "media": 30092, "hotel": 0}
full_init(app, ports[app])

# continues=True: Program will append to the end of existing data
# continues=False: Program will overwrite existing data.
start_test(continues=True)