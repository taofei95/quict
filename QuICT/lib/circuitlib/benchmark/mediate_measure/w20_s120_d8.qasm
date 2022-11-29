OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rz(0.6112224079816593) q[17];
rz(1.8344215471037242) q[12];
rz(1.0432271134262048) q[19];
rz(2.636435163918195) q[7];
rz(4.833276717884174) q[14];
rz(0.9954796087978852) q[8];
rz(4.944621337003283) q[2];
rz(3.882897733076114) q[11];
cx q[10], q[1];
rz(4.005440149356818) q[4];
rz(3.32256744649212) q[9];
cx q[5], q[3];
rz(3.264242371990246) q[6];
cx q[15], q[18];
rz(0.28785384868428876) q[0];
rz(4.174153260209094) q[16];
rz(2.844424999334439) q[13];
cx q[1], q[3];
rz(6.261209962274391) q[0];
rz(4.125484499737656) q[13];
rz(3.13799763672959) q[9];
rz(5.885456218399507) q[19];
rz(1.0058850027505757) q[14];
cx q[2], q[10];
rz(2.5331755577816457) q[7];
rz(4.491153026121014) q[8];
cx q[16], q[6];
rz(5.446922221791971) q[12];
cx q[11], q[17];
rz(5.160824467606073) q[5];
rz(4.567965978700339) q[18];
rz(0.2338894719643892) q[15];
rz(4.2058584898747435) q[4];
rz(3.0392706817727237) q[5];
rz(4.863190570108368) q[8];
rz(0.7301240994524565) q[18];
rz(3.6987101269843694) q[12];
rz(3.3521224405325336) q[17];
cx q[1], q[15];
rz(3.552854499033641) q[10];
cx q[4], q[19];
rz(0.8254395571864309) q[14];
rz(6.272009402695524) q[16];
rz(6.238425083138932) q[9];
cx q[7], q[6];
cx q[0], q[13];
rz(1.516136677666695) q[11];
cx q[2], q[3];
rz(3.028649788523621) q[16];
rz(0.8705237454656096) q[7];
cx q[19], q[4];
rz(1.8147397888128571) q[6];
rz(3.622509321399098) q[9];
cx q[11], q[5];
rz(5.834834675786235) q[2];
rz(5.548806794311105) q[3];
cx q[15], q[1];
rz(4.747896141953102) q[14];
cx q[13], q[17];
rz(3.4288249572479432) q[10];
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
measure q[13] -> c[13];
measure q[14] -> c[14];
measure q[15] -> c[15];
measure q[16] -> c[16];
measure q[17] -> c[17];
measure q[18] -> c[18];
measure q[19] -> c[19];
rz(6.188296623830461) q[12];
rz(3.5805993691300744) q[8];
rz(0.5920835818691218) q[18];
rz(5.39274653513158) q[0];
cx q[7], q[18];
cx q[3], q[10];
rz(2.195991318376484) q[14];
rz(2.583204239089311) q[17];
rz(2.6333613817222665) q[2];
rz(0.7991877740175773) q[1];
rz(1.28444135468388) q[9];
rz(1.4073892822260314) q[12];
rz(5.274224257009472) q[19];
rz(0.6001662499385614) q[0];
cx q[16], q[8];
rz(4.457412387318105) q[13];
rz(4.176655815712763) q[5];
rz(1.0829574223981127) q[15];
cx q[6], q[4];
rz(5.127177941676753) q[11];
cx q[19], q[14];
rz(4.534212334594139) q[17];
rz(5.850848550960839) q[5];
cx q[13], q[6];
cx q[8], q[1];
rz(0.8763855800222737) q[18];
rz(1.3533856027000368) q[11];
rz(5.855671471088467) q[9];
rz(4.31083940128369) q[3];
rz(5.635000639807997) q[10];
rz(4.077750929936712) q[0];
cx q[15], q[16];
rz(3.6927948568099893) q[12];
cx q[4], q[7];
rz(5.8853955436301835) q[2];
rz(4.084342943194392) q[1];
rz(0.6886642657524031) q[12];
rz(2.6959531299867074) q[19];
rz(5.436329041380208) q[10];
rz(0.8360625177615113) q[13];
