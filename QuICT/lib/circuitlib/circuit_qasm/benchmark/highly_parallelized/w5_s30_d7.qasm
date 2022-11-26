OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(5.987253278974372) q[4];
rz(2.3457060022189955) q[3];
cx q[0], q[1];
rz(5.351082704691867) q[2];
rz(0.36547782495436204) q[1];
rz(1.529900759534044) q[0];
rz(1.9903336716345883) q[3];
cx q[4], q[2];
rz(4.107034725075886) q[1];
rz(1.617249657574483) q[4];
rz(4.918110163143253) q[3];
rz(0.1573352424624166) q[0];
rz(1.9915757748954928) q[2];
rz(1.5605369388609922) q[3];
rz(0.7333588516702895) q[1];
rz(1.880868993893149) q[0];
rz(3.0496279151696712) q[4];
rz(6.107875111507902) q[2];
rz(0.6816586836977298) q[4];
cx q[3], q[2];
rz(3.5016871259235893) q[0];
rz(0.6091080730124373) q[1];
cx q[2], q[1];
rz(2.958106325195287) q[0];
rz(0.9247427608493933) q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
