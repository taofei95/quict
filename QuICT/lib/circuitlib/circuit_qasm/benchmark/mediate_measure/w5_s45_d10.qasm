OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c[5];
rz(5.2084810266203405) q[2];
cx q[3], q[0];
rz(2.4328271678535063) q[1];
rz(4.279625524383485) q[4];
rz(2.842027296745074) q[3];
cx q[2], q[4];
rz(2.8052872008854046) q[1];
rz(4.6030635371324715) q[0];
rz(1.3543114114367092) q[1];
rz(6.084313821514307) q[3];
rz(3.046156303626955) q[2];
rz(2.2699816537310507) q[4];
rz(0.4101390773073695) q[0];
rz(0.3846003785005375) q[1];
rz(2.004428786183025) q[0];
rz(5.41316974783307) q[3];
rz(4.07479450989402) q[2];
rz(3.8845687009902194) q[4];
rz(6.11822815289252) q[0];
rz(4.8269379152006895) q[3];
rz(5.046906923516995) q[4];
rz(1.980156293023012) q[1];
rz(1.3551325566365735) q[2];
rz(3.0952426623379705) q[2];
rz(5.164221949458942) q[3];
rz(2.230085257774881) q[0];
rz(2.5167594896220886) q[4];
rz(2.8804970078185863) q[1];
rz(2.3027747438584463) q[2];
rz(0.7916481884674083) q[0];
rz(1.3105580823386784) q[3];
rz(0.1657640593927584) q[1];
rz(1.9470300488808074) q[4];
rz(3.800300795595046) q[0];
rz(1.497677531139261) q[3];
rz(4.1579381294236315) q[4];
cx q[1], q[2];
rz(5.805948490856477) q[0];
rz(1.383031173465416) q[4];
rz(2.566440352658395) q[2];
rz(0.7542798944806188) q[1];
rz(5.007941628416046) q[3];
rz(0.9117599266617709) q[4];
rz(0.5599197875529645) q[0];
rz(4.5608519808021315) q[1];
