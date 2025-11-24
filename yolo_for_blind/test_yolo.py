import time, mindspore_lite as mslite

start = time.time()
ctx = mslite.Context()
ctx.target=["Ascend"]
m = mslite.Model()
m.build_from_file("/home/HwHiAiUser/work/hejunhao/yolo_for_blind/yolov8s_wotr.mindir", mslite.ModelType.MINDIR, ctx)
print("load time", time.time() - start)
