OPENQASM 2.0;
include "qelib1.inc";
qreg q[50];
creg c[50];
x q[0];
x q[1];
x q[2];
x q[3];
x q[4];
x q[5];
x q[7];
x q[8];
x q[9];
x q[10];
x q[12];
x q[13];
x q[14];
x q[15];
x q[16];
x q[17];
x q[18];
x q[19];
x q[20];
x q[21];
x q[23];
x q[26];
x q[27];
x q[29];
x q[31];
x q[32];
x q[33];
x q[35];
x q[37];
x q[38];
x q[39];
x q[40];
x q[41];
x q[42];
x q[43];
x q[45];
x q[0];
h q[0];
rxx(0.0231894850730896) q[0], q[49];
rxx(0.43382585048675537) q[1], q[49];
rxx(0.38885581493377686) q[2], q[49];
rxx(0.183332622051239) q[3], q[49];
rxx(0.39345866441726685) q[4], q[49];
rxx(0.6160181164741516) q[5], q[49];
rxx(0.8980556726455688) q[6], q[49];
rxx(0.3773653507232666) q[7], q[49];
rxx(0.36845070123672485) q[8], q[49];
rxx(0.2671423554420471) q[9], q[49];
rxx(0.8345788717269897) q[10], q[49];
rxx(0.715725839138031) q[11], q[49];
rxx(0.03432673215866089) q[12], q[49];
rxx(0.495688259601593) q[13], q[49];
rxx(0.3339017629623413) q[14], q[49];
rxx(0.2071695327758789) q[15], q[49];
rxx(0.01880091428756714) q[16], q[49];
rxx(0.8876044154167175) q[17], q[49];
rxx(0.9460865259170532) q[18], q[49];
rxx(0.23233461380004883) q[19], q[49];
rxx(0.9813522696495056) q[20], q[49];
rxx(0.7539567351341248) q[21], q[49];
rxx(0.8899063467979431) q[22], q[49];
rxx(0.7160829305648804) q[23], q[49];
rxx(0.4057914614677429) q[24], q[49];
rxx(0.5641207695007324) q[25], q[49];
rxx(0.31237727403640747) q[26], q[49];
rxx(0.4548875093460083) q[27], q[49];
rxx(0.3915167450904846) q[28], q[49];
rxx(0.4576558470726013) q[29], q[49];
rxx(0.3800595998764038) q[30], q[49];
rxx(0.3130451440811157) q[31], q[49];
rxx(0.07727915048599243) q[32], q[49];
rxx(0.18030214309692383) q[33], q[49];
rxx(0.0012921690940856934) q[34], q[49];
rxx(0.14192217588424683) q[35], q[49];
rxx(0.4792870283126831) q[36], q[49];
rxx(0.7976175546646118) q[37], q[49];
rxx(0.26892513036727905) q[38], q[49];
rxx(0.66403728723526) q[39], q[49];
rxx(0.15914446115493774) q[40], q[49];
rxx(0.33896392583847046) q[41], q[49];
rxx(0.9790598750114441) q[42], q[49];
rxx(0.5795366764068604) q[43], q[49];
rxx(0.684565544128418) q[44], q[49];
rxx(0.0890464186668396) q[45], q[49];
rxx(0.6866880655288696) q[46], q[49];
rxx(0.10645538568496704) q[47], q[49];
rxx(0.2659945487976074) q[48], q[49];
rzx(0.8622496724128723) q[0], q[49];
rzx(0.019696354866027832) q[1], q[49];
rzx(0.6245175004005432) q[2], q[49];
rzx(0.4229588508605957) q[3], q[49];
rzx(0.06406188011169434) q[4], q[49];
rzx(0.8169119954109192) q[5], q[49];
rzx(0.7869608402252197) q[6], q[49];
rzx(0.9747675657272339) q[7], q[49];
rzx(0.9864874482154846) q[8], q[49];
rzx(0.4520282745361328) q[9], q[49];
rzx(0.004773259162902832) q[10], q[49];
rzx(0.2901275157928467) q[11], q[49];
rzx(0.48509687185287476) q[12], q[49];
rzx(0.07896757125854492) q[13], q[49];
rzx(0.32296115159988403) q[14], q[49];
rzx(0.7855817079544067) q[15], q[49];
rzx(0.5638002753257751) q[16], q[49];
rzx(0.6075443625450134) q[17], q[49];
rzx(0.9312153458595276) q[18], q[49];
rzx(0.7688433527946472) q[19], q[49];
rzx(0.5515389442443848) q[20], q[49];
rzx(0.0509982705116272) q[21], q[49];
rzx(0.4816058874130249) q[22], q[49];
rzx(0.4867206811904907) q[23], q[49];
rzx(0.11522585153579712) q[24], q[49];
rzx(0.1667722463607788) q[25], q[49];
rzx(0.715725839138031) q[26], q[49];
rzx(0.3834603428840637) q[27], q[49];
rzx(0.5499125719070435) q[28], q[49];
rzx(0.4424411654472351) q[29], q[49];
rzx(0.7884268760681152) q[30], q[49];
rzx(0.7712355852127075) q[31], q[49];
rzx(0.7411074638366699) q[32], q[49];
rzx(0.5912990570068359) q[33], q[49];
rzx(0.6953386664390564) q[34], q[49];
rzx(0.8800061345100403) q[35], q[49];
rzx(0.6177265048027039) q[36], q[49];
rzx(0.8008987307548523) q[37], q[49];
rzx(0.9057193398475647) q[38], q[49];
rzx(0.3270684480667114) q[39], q[49];
rzx(0.9316422343254089) q[40], q[49];
rzx(0.29741454124450684) q[41], q[49];
rzx(0.7477120757102966) q[42], q[49];
rzx(0.3717331886291504) q[43], q[49];
rzx(0.922564685344696) q[44], q[49];
rzx(0.6689146757125854) q[45], q[49];
rzx(0.9615466594696045) q[46], q[49];
rzx(0.8599694967269897) q[47], q[49];
rzx(0.8497874140739441) q[48], q[49];
h q[0];