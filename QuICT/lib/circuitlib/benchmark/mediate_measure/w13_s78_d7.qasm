OPENQASM 2.0;
include "qelib1.inc";
qreg q[13];
creg c[13];
rz(4.02911836909868) q[12];
rz(4.593669653893519) q[9];
rz(4.386145985557201) q[0];
rz(5.385939582462522) q[5];
rz(5.404132934060453) q[3];
rz(1.1985141545438827) q[2];
rz(4.327080696956221) q[1];
rz(5.904488458325211) q[7];
cx q[6], q[8];
cx q[10], q[4];
rz(5.401663509549131) q[11];
cx q[3], q[2];
rz(4.418933897632797) q[4];
rz(1.4738966007846757) q[10];
rz(0.3177456682361146) q[1];
rz(3.5167421741586913) q[8];
cx q[7], q[0];
rz(2.6991375022473036) q[9];
rz(4.191922136180219) q[6];
rz(1.3913773973137564) q[12];
rz(1.8235661324463053) q[11];
rz(3.230823569354454) q[5];
cx q[6], q[12];
rz(5.796475031773636) q[5];
cx q[4], q[7];
rz(4.69480042712098) q[2];
rz(2.838367519531729) q[1];
cx q[11], q[8];
rz(2.440253602705851) q[0];
rz(3.1672457515023624) q[3];
rz(3.4575975729930275) q[10];
rz(5.267101667014318) q[9];
rz(5.864728283542231) q[4];
rz(1.4914587851592236) q[2];
rz(3.19195046494382) q[1];
rz(3.3132195716563273) q[11];
rz(1.3927632770763614) q[8];
rz(0.9179215803573811) q[5];
cx q[0], q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];
measure q[4] -> c[4];
measure q[5] -> c[5];
measure q[6] -> c[6];
measure q[7] -> c[7];
measure q[8] -> c[8];
measure q[9] -> c[9];
measure q[10] -> c[10];
measure q[11] -> c[11];
measure q[12] -> c[12];
rz(4.190175641107174) q[7];
rz(4.893426013989055) q[9];
rz(4.618167969819711) q[6];
rz(4.329450938556914) q[12];
rz(0.9703941486518238) q[10];
rz(3.8583217883333605) q[4];
cx q[5], q[6];
rz(1.7128406176732378) q[0];
rz(5.9221702937391605) q[8];
rz(3.8116205782887995) q[9];
cx q[1], q[10];
rz(0.037978942313814816) q[2];
rz(2.0040191116294976) q[3];
rz(0.3171884575474054) q[12];
rz(4.62766624346155) q[7];
rz(5.399592377469478) q[11];
rz(3.217041400577253) q[3];
rz(2.3436325349290685) q[2];
rz(3.275656566995924) q[6];
rz(3.3915511665458267) q[9];
rz(3.9247453029153068) q[11];
rz(1.6531631403331026) q[12];
cx q[4], q[0];
rz(1.126326605579788) q[1];
cx q[7], q[5];
rz(0.4107365343027368) q[8];