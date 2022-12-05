OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(0.25519301950628337) q[0];
rz(6.037664115557289) q[2];
rz(5.115958241045053) q[1];
rz(3.707184550727427) q[2];
rz(0.30786190227399185) q[1];
rz(2.575329522366595) q[0];
rz(6.267329718167377) q[2];
rz(2.3083822205862874) q[1];
rz(3.832292249276608) q[0];
rz(0.3093918880789226) q[1];
rz(0.518173007881506) q[0];
rz(1.307321469846023) q[2];
cx q[2], q[1];
rz(4.722512508958906) q[0];
rz(5.596677068605878) q[2];
cx q[1], q[0];
rz(4.854033694527681) q[0];
cx q[2], q[1];
rz(1.9071282515490267) q[2];
rz(1.4928380978454785) q[1];
rz(5.966171474498183) q[0];
rz(3.9043660835574197) q[0];
cx q[1], q[2];
cx q[1], q[2];
rz(5.16021853544152) q[0];
rz(1.9602341352196981) q[0];
rz(4.718279943396037) q[1];
rz(0.27089722158449414) q[2];
rz(1.9527320717852625) q[1];
rz(0.624324861945354) q[2];
rz(0.6868986090558492) q[0];
rz(3.0238325495906935) q[1];
cx q[2], q[0];
rz(1.4365251019142602) q[1];
rz(1.994800387012289) q[2];
rz(0.16235476642954422) q[0];
cx q[2], q[0];
rz(3.14517602957528) q[1];
cx q[0], q[2];
rz(2.9324129484395725) q[1];
rz(3.364035274324973) q[1];
cx q[2], q[0];
cx q[0], q[1];
rz(0.39299555066032127) q[2];
rz(4.042147227647373) q[1];
rz(4.228635385347611) q[2];
rz(3.6373230025780607) q[0];
rz(1.4322179357173472) q[0];
rz(1.5977142004049194) q[1];
rz(1.759661529328972) q[2];
cx q[1], q[2];
rz(5.0236955741435665) q[0];
rz(2.246315639536869) q[1];
rz(1.6345082716278116) q[2];
rz(1.4127518582388765) q[0];
cx q[1], q[2];
rz(0.2517314528214945) q[0];
rz(0.6533598824470426) q[1];
rz(5.963087648916834) q[2];
rz(0.3117702766873349) q[0];
rz(3.7679362705393467) q[0];
rz(5.060178958200939) q[2];
rz(3.9330745040618167) q[1];
rz(4.108163416092013) q[2];
rz(3.687758253150884) q[0];
rz(2.3043069590417016) q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
