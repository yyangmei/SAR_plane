单类模型测试：

             demo_boat.py  只检测船

             demo_airplane.py 直接只检测飞机（把大图裁小直接检测飞机）

             demo_plane.py 只检测飞机（先检测机场再在机场上检测飞机，只输出飞机结果）

两类一起测试：

             demo_ship_plane.py 先检测船，然后再检测飞机，不检测机场


             demo_ALL.py 先检测船，然后在检测机场，在机场的基础上检测飞机
             
             demo_shipplane_nms.py 先检测船再检测飞机，不检测机场，两类放在一起做nms