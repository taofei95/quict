OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
rz(6.061865052902968) q[1];
rz(4.968421312379448) q[2];
rz(2.769460723349261) q[0];
rz(1.3631394456215369) q[2];
rz(2.2477182462396788) q[1];
rz(2.364429584198685) q[0];
rz(0.938649549423779) q[1];
cx q[2], q[0];
rz(1.0763566233390671) q[0];
rz(1.4210020770110794) q[1];
rz(4.53344651171654) q[2];
cx q[0], q[1];
rz(5.062650262912164) q[2];
rz(3.960503189164394) q[1];
rz(3.9991842244900986) q[2];
rz(0.6231218953864757) q[0];
rz(2.2876386648498057) q[0];
rz(4.828237295894127) q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
rz(1.6751467094609833) q[1];
rz(4.496841728099425) q[2];
rz(6.036157347941935) q[0];
rz(4.332525302453451) q[1];
cx q[0], q[1];
rz(5.920402586792106) q[2];
cx q[2], q[1];
rz(3.617635123293602) q[0];
rz(0.7684380321412884) q[1];
rz(5.326587610775414) q[2];
rz(5.211563897892083) q[0];
rz(5.418924936377858) q[2];
rz(1.4876364980886583) q[1];
rz(2.3136777180472916) q[0];
rz(2.701673591489764) q[1];
