OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
creg c[20];
rz(3.4536770075698957) q[8];
rz(6.188488067256216) q[18];
cx q[19], q[1];
cx q[5], q[12];
rz(0.7331452953154728) q[3];
rz(0.5234670811309897) q[2];
rz(5.471610215100995) q[4];
rz(0.20142564595906637) q[16];
rz(4.037676465452967) q[14];
rz(4.691210743146447) q[7];
rz(2.129226466575112) q[9];
rz(1.6087647376502507) q[0];
rz(1.0567039717295708) q[11];
rz(4.46230031880084) q[17];
rz(5.035525019890415) q[10];
cx q[6], q[13];
rz(6.0226927389045) q[15];
rz(0.3183735742959698) q[8];
rz(5.954232531507755) q[17];
cx q[9], q[4];
rz(1.0739381306704052) q[5];
rz(5.135033515541827) q[0];
cx q[11], q[3];
rz(3.8409335387872017) q[14];
rz(1.1819009394072892) q[7];
rz(5.768109669426808) q[19];
rz(4.2207056849128275) q[18];
rz(5.020226355965112) q[13];
rz(6.10610319354129) q[2];
cx q[16], q[6];
rz(2.6950409853046047) q[1];
cx q[12], q[15];
rz(3.465469267044344) q[10];
rz(2.833294904964959) q[17];
rz(4.901426035304773) q[14];
rz(0.917947109695196) q[7];
rz(0.6652997485730415) q[8];
cx q[4], q[18];
rz(2.6824605046339043) q[9];
rz(5.626528996562623) q[15];
rz(4.412769170626273) q[13];
rz(2.3107481590546017) q[6];
rz(5.748022453937077) q[3];
rz(5.797278449039254) q[10];
rz(0.5053344020985823) q[11];
rz(3.61014805499695) q[0];
rz(2.3825877778398876) q[12];
rz(1.3127049474532777) q[19];
rz(3.1487868285570255) q[16];
rz(0.6286505167337352) q[5];
cx q[1], q[2];
rz(4.616107852590477) q[12];
rz(5.343615975893195) q[1];
rz(4.300550467065958) q[10];
cx q[2], q[0];
cx q[16], q[11];
rz(0.21934298195407329) q[3];
rz(3.4269368380995076) q[13];
rz(1.3271924836270397) q[14];
rz(5.351212726244292) q[7];
rz(3.1904979808484217) q[8];
rz(4.855658235552071) q[4];
cx q[9], q[15];
rz(4.9588915900015955) q[6];
rz(3.223287220415416) q[17];
cx q[18], q[5];
rz(2.64304102056085) q[19];
rz(5.671786092285421) q[4];
cx q[9], q[12];
rz(3.438227776425295) q[15];
rz(4.116811767734693) q[7];
rz(2.5512303414252417) q[11];
rz(5.434785567635584) q[13];
rz(0.7625170308784444) q[10];
rz(1.7715153104258057) q[19];
cx q[8], q[5];
rz(2.108326765803266) q[16];
rz(5.236258182675861) q[3];
rz(5.324611360655056) q[0];
rz(5.093330127293139) q[17];
rz(1.0001789728893382) q[18];
rz(0.2885081002421214) q[6];
rz(3.6415240214138103) q[14];
rz(1.3434504334588984) q[1];
rz(6.264899784769825) q[2];
rz(1.7206404505810482) q[9];
rz(1.833780089868726) q[12];
rz(0.22123007262280794) q[17];
rz(0.5302567560264032) q[3];
rz(1.4106822709988518) q[18];
rz(3.3356373943566915) q[16];
rz(3.81958615270087) q[11];
cx q[1], q[2];
cx q[19], q[0];
rz(3.5834386512903804) q[5];
rz(3.4682753340793826) q[14];
rz(0.305354611567832) q[4];
rz(1.483689402321176) q[7];
rz(0.02585187158507159) q[15];
rz(0.9408703785276759) q[10];
rz(0.5236547644901194) q[8];
rz(0.4284745934697885) q[6];
rz(0.2736427589154667) q[13];
rz(4.256984645936895) q[1];
cx q[19], q[5];
rz(5.643614873392471) q[14];
rz(3.13283180795542) q[0];
rz(2.8754993644697526) q[13];
rz(5.2031876955722876) q[17];
rz(1.244220088796149) q[7];
cx q[4], q[12];
rz(1.6170058319436662) q[10];
rz(0.5360709876751196) q[11];
rz(3.7086485063345447) q[2];
rz(0.9038764546927351) q[3];
rz(5.481333400794185) q[8];
cx q[18], q[9];
cx q[15], q[6];
rz(6.117569919947319) q[16];
rz(4.136420132803241) q[14];
rz(6.217604788996241) q[17];
rz(3.0847572734893958) q[10];
rz(4.673759557769488) q[16];
rz(0.19182855060662007) q[3];
cx q[15], q[1];
rz(5.048361870257876) q[5];
rz(1.7013403148921318) q[12];
rz(3.7431176492138225) q[2];
rz(2.34073961783994) q[8];
rz(1.3088707989155504) q[11];
rz(0.9813441760022069) q[19];
rz(0.6523364189564341) q[0];
rz(3.2322652843268993) q[4];
rz(3.134054342382582) q[9];
rz(0.5773934223351581) q[6];
rz(1.2905761860102867) q[7];
rz(4.7343318964658305) q[13];
rz(0.20843702817097998) q[18];
cx q[15], q[0];
rz(1.7984481345981762) q[1];
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
