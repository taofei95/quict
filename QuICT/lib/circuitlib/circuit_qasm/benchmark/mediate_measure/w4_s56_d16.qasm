OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
rz(0.5633632868813717) q[1];
rz(1.105191242222149) q[0];
rz(5.667874979661133) q[3];
rz(5.305427562280316) q[2];
cx q[3], q[1];
cx q[0], q[2];
rz(0.5033493733679724) q[2];
rz(4.229313658250528) q[0];
rz(5.41481208561934) q[3];
rz(5.755524898005035) q[1];
rz(4.0051186163185095) q[3];
cx q[1], q[0];
rz(0.4662123606873974) q[2];
rz(0.41930636932399495) q[1];
rz(2.5387976954119815) q[3];
cx q[2], q[0];
cx q[3], q[1];
rz(2.278514505235732) q[2];
rz(5.765282492355169) q[0];
rz(3.8852595142842032) q[1];
rz(1.130513975902922) q[0];
rz(5.232428700183255) q[3];
rz(2.7815374504671437) q[2];
rz(0.6832073284346002) q[3];
rz(1.4141082135827772) q[1];
rz(2.765829674250676) q[0];
rz(5.783604667370429) q[2];
rz(3.580311972631702) q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
rz(6.247291274275095) q[1];
rz(0.8449676493633314) q[3];
rz(3.9681170910203765) q[0];
rz(1.2803947074589137) q[2];
rz(5.749575696890008) q[3];
rz(5.963265284825038) q[1];
rz(6.214740947063132) q[0];
rz(4.4944305918663625) q[2];
cx q[0], q[3];
rz(3.327416889780772) q[1];
rz(0.5835126327117609) q[1];
rz(4.126792261470923) q[0];
rz(0.9918361763614141) q[2];
rz(2.370685070388167) q[3];
rz(2.274237589393723) q[0];
cx q[2], q[3];
rz(2.4212002508855672) q[1];
rz(3.7466814359000216) q[2];
rz(0.40696349218158273) q[3];
rz(4.774262256324079) q[1];
rz(4.353505091885946) q[0];
rz(2.602907874236167) q[1];
rz(0.5892108922789704) q[0];
cx q[3], q[2];
