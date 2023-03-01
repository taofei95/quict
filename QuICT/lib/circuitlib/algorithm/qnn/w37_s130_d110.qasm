OPENQASM 2.0;
include "qelib1.inc";
qreg q[37];
creg c[37];
x q[1];
x q[5];
x q[6];
x q[8];
x q[9];
x q[12];
x q[13];
x q[14];
x q[17];
x q[18];
x q[20];
x q[21];
x q[22];
x q[23];
x q[27];
x q[28];
x q[30];
x q[31];
x q[34];
x q[0];
h q[0];
rxx(0.43926966190338135) q[0], q[36];
rxx(0.7117254137992859) q[1], q[36];
rxx(0.7289010882377625) q[2], q[36];
rxx(0.9568814039230347) q[3], q[36];
rxx(0.8908405900001526) q[4], q[36];
rxx(0.2791866064071655) q[5], q[36];
rxx(0.7495810389518738) q[6], q[36];
rxx(0.1681951880455017) q[7], q[36];
rxx(0.8292604088783264) q[8], q[36];
rxx(0.27475517988204956) q[9], q[36];
rxx(0.11656677722930908) q[10], q[36];
rxx(0.2383221983909607) q[11], q[36];
rxx(0.24378693103790283) q[12], q[36];
rxx(0.2666170001029968) q[13], q[36];
rxx(0.08800864219665527) q[14], q[36];
rxx(0.51158607006073) q[15], q[36];
rxx(0.5378339886665344) q[16], q[36];
rxx(0.0739404559135437) q[17], q[36];
rxx(0.9778372645378113) q[18], q[36];
rxx(0.2034408450126648) q[19], q[36];
rxx(0.016119539737701416) q[20], q[36];
rxx(0.11588901281356812) q[21], q[36];
rxx(0.9944096207618713) q[22], q[36];
rxx(0.5544820427894592) q[23], q[36];
rxx(0.583335816860199) q[24], q[36];
rxx(0.48481374979019165) q[25], q[36];
rxx(0.8716122508049011) q[26], q[36];
rxx(0.6207879185676575) q[27], q[36];
rxx(0.8230307102203369) q[28], q[36];
rxx(0.8369712233543396) q[29], q[36];
rxx(0.9775185585021973) q[30], q[36];
rxx(0.3054124116897583) q[31], q[36];
rxx(0.49386316537857056) q[32], q[36];
rxx(0.6859082579612732) q[33], q[36];
rxx(0.624484121799469) q[34], q[36];
rxx(0.7933400869369507) q[35], q[36];
ryy(0.18811285495758057) q[0], q[36];
ryy(0.8883539438247681) q[1], q[36];
ryy(0.8831125497817993) q[2], q[36];
ryy(0.42818933725357056) q[3], q[36];
ryy(0.9335808157920837) q[4], q[36];
ryy(0.3199170231819153) q[5], q[36];
ryy(0.9284586310386658) q[6], q[36];
ryy(0.013990998268127441) q[7], q[36];
ryy(0.65382319688797) q[8], q[36];
ryy(0.4618456959724426) q[9], q[36];
ryy(0.695191502571106) q[10], q[36];
ryy(0.734481930732727) q[11], q[36];
ryy(0.49399954080581665) q[12], q[36];
ryy(0.3411175608634949) q[13], q[36];
ryy(0.7559918761253357) q[14], q[36];
ryy(0.024175524711608887) q[15], q[36];
ryy(0.7728071808815002) q[16], q[36];
ryy(0.047077834606170654) q[17], q[36];
ryy(0.41840660572052) q[18], q[36];
ryy(0.8307532072067261) q[19], q[36];
ryy(0.6052560806274414) q[20], q[36];
ryy(0.4950469732284546) q[21], q[36];
ryy(0.0947960615158081) q[22], q[36];
ryy(0.12633979320526123) q[23], q[36];
ryy(0.2966804504394531) q[24], q[36];
ryy(0.6502265930175781) q[25], q[36];
ryy(0.025502145290374756) q[26], q[36];
ryy(0.9535624980926514) q[27], q[36];
ryy(0.7919332981109619) q[28], q[36];
ryy(0.16847443580627441) q[29], q[36];
ryy(0.6757118701934814) q[30], q[36];
ryy(0.6993317008018494) q[31], q[36];
ryy(0.9492158889770508) q[32], q[36];
ryy(0.5204731225967407) q[33], q[36];
ryy(0.139448344707489) q[34], q[36];
ryy(0.011517345905303955) q[35], q[36];
rzx(0.7757032513618469) q[0], q[36];
rzx(0.6014034748077393) q[1], q[36];
rzx(0.4361507296562195) q[2], q[36];
rzx(0.6918910145759583) q[3], q[36];
rzx(0.5587146878242493) q[4], q[36];
rzx(0.08080726861953735) q[5], q[36];
rzx(0.2122604250907898) q[6], q[36];
rzx(0.7806897163391113) q[7], q[36];
rzx(0.2647048234939575) q[8], q[36];
rzx(0.546203076839447) q[9], q[36];
rzx(0.7327220439910889) q[10], q[36];
rzx(0.8312007188796997) q[11], q[36];
rzx(0.8092151880264282) q[12], q[36];
rzx(0.5084642171859741) q[13], q[36];
rzx(0.3646976351737976) q[14], q[36];
rzx(0.8348162174224854) q[15], q[36];
rzx(0.41291922330856323) q[16], q[36];
rzx(0.09387117624282837) q[17], q[36];
rzx(0.8123677968978882) q[18], q[36];
rzx(0.21682780981063843) q[19], q[36];
rzx(0.26575541496276855) q[20], q[36];
rzx(0.9447059035301208) q[21], q[36];
rzx(0.6601765751838684) q[22], q[36];
rzx(0.5522696375846863) q[23], q[36];
rzx(0.7826545834541321) q[24], q[36];
rzx(0.8162859678268433) q[25], q[36];
rzx(0.2830004096031189) q[26], q[36];
rzx(0.4853396415710449) q[27], q[36];
rzx(0.2707492709159851) q[28], q[36];
rzx(0.7619187831878662) q[29], q[36];
rzx(0.6048544645309448) q[30], q[36];
rzx(0.21513152122497559) q[31], q[36];
rzx(0.5032123923301697) q[32], q[36];
rzx(0.2518230080604553) q[33], q[36];
rzx(0.5604069232940674) q[34], q[36];
rzx(0.6574825644493103) q[35], q[36];
h q[0];