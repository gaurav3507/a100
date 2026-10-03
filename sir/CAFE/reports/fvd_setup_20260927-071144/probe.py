
import json, os, sys
out = {}
try:
    import tensorflow as tf, tensorflow_hub as hub, tensorflow_gan as tfgan, tensorflow_probability as tfp
    import cv2, numpy, pandas
    out["versions"] = {"tensorflow": tf.__version__, "tensorflow_hub": hub.__version__,
                       "tensorflow_gan": getattr(tfgan, "__version__", "?"), "tensorflow_probability": tfp.__version__,
                       "cv2": cv2.__version__, "numpy": numpy.__version__, "pandas": pandas.__version__}
    out["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    out["resolved"] = hub.resolve(sys.argv[1])
    tf1 = tf.compat.v1
    tf1.disable_v2_behavior()
    with tf1.Graph().as_default():
        x = tf1.placeholder(tf1.float32, [16, 15, 224, 224, 3])
        m = hub.Module(sys.argv[1], name="probe_module")
        m(x)
        t = tf1.get_default_graph().get_tensor_by_name("probe_module_apply_default/RGB/inception_i3d/Mean:0")
        cfg = tf1.ConfigProto()
        cfg.gpu_options.allow_growth = True
        with tf1.Session(config=cfg) as s:
            s.run(tf1.global_variables_initializer())
            s.run(tf1.tables_initializer())
            v = s.run(t, {x: numpy.zeros([16, 15, 224, 224, 3], "float32")})
    out["embedding_shape"] = list(v.shape)
    out["embedding_finite"] = bool(numpy.isfinite(v).all())
    out["n_variables"] = len(m.variables)
except Exception as e:
    out["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
print("PROBE_JSON " + json.dumps(out))
