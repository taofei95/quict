OPENQASM 2.0;
include "qelib1.inc";
qreg q[82];
creg c[82];
x q[0];
x q[3];
x q[7];
x q[8];
x q[9];
x q[11];
x q[12];
x q[13];
x q[14];
x q[15];
x q[16];
x q[20];
x q[21];
x q[24];
x q[25];
x q[26];
x q[28];
x q[32];
x q[39];
x q[40];
x q[41];
x q[42];
x q[44];
x q[50];
x q[51];
x q[52];
x q[54];
x q[55];
x q[58];
x q[61];
x q[62];
x q[65];
x q[66];
x q[67];
x q[68];
x q[69];
x q[72];
x q[73];
x q[75];
x q[76];
x q[77];
x q[78];
x q[0];
h q[0];
rxx(0.5857064723968506) q[0], q[81];
rxx(0.5047641396522522) q[1], q[81];
rxx(0.4353579878807068) q[2], q[81];
rxx(0.6385848522186279) q[3], q[81];
rxx(0.031200528144836426) q[4], q[81];
rxx(0.6352231502532959) q[5], q[81];
rxx(0.5585932731628418) q[6], q[81];
rxx(0.07749444246292114) q[7], q[81];
rxx(0.4374825358390808) q[8], q[81];
rxx(0.7920902967453003) q[9], q[81];
rxx(0.3837931752204895) q[10], q[81];
rxx(0.2561858892440796) q[11], q[81];
rxx(0.4162583351135254) q[12], q[81];
rxx(0.7049850821495056) q[13], q[81];
rxx(0.681104838848114) q[14], q[81];
rxx(0.1506211757659912) q[15], q[81];
rxx(0.1374068260192871) q[16], q[81];
rxx(0.6464861631393433) q[17], q[81];
rxx(0.6170086860656738) q[18], q[81];
rxx(0.10751688480377197) q[19], q[81];
rxx(0.7150081992149353) q[20], q[81];
rxx(0.7648976445198059) q[21], q[81];
rxx(0.3072795867919922) q[22], q[81];
rxx(0.8041247129440308) q[23], q[81];
rxx(0.8172578811645508) q[24], q[81];
rxx(0.6898828744888306) q[25], q[81];
rxx(0.9449101090431213) q[26], q[81];
rxx(0.13187628984451294) q[27], q[81];
rxx(0.8354941606521606) q[28], q[81];
rxx(0.8438165187835693) q[29], q[81];
rxx(0.6932749152183533) q[30], q[81];
rxx(0.26878678798675537) q[31], q[81];
rxx(0.294258713722229) q[32], q[81];
rxx(0.8090174198150635) q[33], q[81];
rxx(0.04640132188796997) q[34], q[81];
rxx(0.5987306237220764) q[35], q[81];
rxx(0.1776828169822693) q[36], q[81];
rxx(0.8938273787498474) q[37], q[81];
rxx(0.20943397283554077) q[38], q[81];
rxx(0.5690271258354187) q[39], q[81];
rxx(0.18646860122680664) q[40], q[81];
rxx(0.18690335750579834) q[41], q[81];
rxx(0.5585049390792847) q[42], q[81];
rxx(0.6785769462585449) q[43], q[81];
rxx(0.9085832238197327) q[44], q[81];
rxx(0.4840473532676697) q[45], q[81];
rxx(0.08900558948516846) q[46], q[81];
rxx(0.7483450770378113) q[47], q[81];
rxx(0.3072469234466553) q[48], q[81];
rxx(0.4291675090789795) q[49], q[81];
rxx(0.634615421295166) q[50], q[81];
rxx(0.3255969285964966) q[51], q[81];
rxx(0.7252099514007568) q[52], q[81];
rxx(0.2690652012825012) q[53], q[81];
rxx(0.3670966625213623) q[54], q[81];
rxx(0.4728902578353882) q[55], q[81];
rxx(0.7661353945732117) q[56], q[81];
rxx(0.5888121724128723) q[57], q[81];
rxx(0.11215263605117798) q[58], q[81];
rxx(0.9339253902435303) q[59], q[81];
rxx(0.11086457967758179) q[60], q[81];
rxx(0.12534981966018677) q[61], q[81];
rxx(0.2929745316505432) q[62], q[81];
rxx(0.08438992500305176) q[63], q[81];
rxx(0.5037773251533508) q[64], q[81];
rxx(0.5795257091522217) q[65], q[81];
rxx(0.7784122228622437) q[66], q[81];
rxx(0.02242577075958252) q[67], q[81];
rxx(0.9178016185760498) q[68], q[81];
rxx(0.28034430742263794) q[69], q[81];
rxx(0.20402199029922485) q[70], q[81];
rxx(0.3452569842338562) q[71], q[81];
rxx(0.6692755818367004) q[72], q[81];
rxx(0.5057985186576843) q[73], q[81];
rxx(0.41481125354766846) q[74], q[81];
rxx(0.5022878050804138) q[75], q[81];
rxx(0.14656537771224976) q[76], q[81];
rxx(0.23832917213439941) q[77], q[81];
rxx(0.7501150369644165) q[78], q[81];
rxx(0.11978363990783691) q[79], q[81];
rxx(0.9915801286697388) q[80], q[81];
rzz(0.6972492337226868) q[0], q[81];
rzz(0.1647847294807434) q[1], q[81];
rzz(0.38079339265823364) q[2], q[81];
rzz(0.9708961248397827) q[3], q[81];
rzz(0.26925235986709595) q[4], q[81];
rzz(0.6288373470306396) q[5], q[81];
rzz(0.6632714867591858) q[6], q[81];
rzz(0.5313045978546143) q[7], q[81];
rzz(0.8279464840888977) q[8], q[81];
rzz(0.6198776960372925) q[9], q[81];
rzz(0.34726983308792114) q[10], q[81];
rzz(0.6451529264450073) q[11], q[81];
rzz(0.005583345890045166) q[12], q[81];
rzz(0.8689923286437988) q[13], q[81];
rzz(0.8452633619308472) q[14], q[81];
rzz(0.9898091554641724) q[15], q[81];
rzz(0.36100393533706665) q[16], q[81];
rzz(0.9167135953903198) q[17], q[81];
rzz(0.797132670879364) q[18], q[81];
rzz(0.0026824474334716797) q[19], q[81];
rzz(0.2384638786315918) q[20], q[81];
rzz(0.2072739601135254) q[21], q[81];
rzz(0.6896697878837585) q[22], q[81];
rzz(0.005786299705505371) q[23], q[81];
rzz(0.7603238821029663) q[24], q[81];
rzz(0.7977352142333984) q[25], q[81];
rzz(0.564197301864624) q[26], q[81];
rzz(0.5598316192626953) q[27], q[81];
rzz(0.6060384511947632) q[28], q[81];
rzz(0.8257103562355042) q[29], q[81];
rzz(0.6043879985809326) q[30], q[81];
rzz(0.7211486101150513) q[31], q[81];
rzz(0.4769149422645569) q[32], q[81];
rzz(0.8729802370071411) q[33], q[81];
rzz(0.4707726836204529) q[34], q[81];
rzz(0.4702720046043396) q[35], q[81];
rzz(0.1136636734008789) q[36], q[81];
rzz(0.6521439552307129) q[37], q[81];
rzz(0.9890879988670349) q[38], q[81];
rzz(0.2619563937187195) q[39], q[81];
rzz(0.9531775712966919) q[40], q[81];
rzz(0.8558573722839355) q[41], q[81];
rzz(0.8447892069816589) q[42], q[81];
rzz(0.09789752960205078) q[43], q[81];
rzz(0.15286272764205933) q[44], q[81];
rzz(0.7426693439483643) q[45], q[81];
rzz(0.43729329109191895) q[46], q[81];
rzz(0.8845008015632629) q[47], q[81];
rzz(0.6638210415840149) q[48], q[81];
rzz(0.03972649574279785) q[49], q[81];
rzz(0.5113972425460815) q[50], q[81];
rzz(0.9754071831703186) q[51], q[81];
rzz(0.87849360704422) q[52], q[81];
rzz(0.7143722772598267) q[53], q[81];
rzz(0.1074184775352478) q[54], q[81];
rzz(0.5048173666000366) q[55], q[81];
rzz(0.24851304292678833) q[56], q[81];
rzz(0.22927069664001465) q[57], q[81];
rzz(0.5409631133079529) q[58], q[81];
rzz(0.9159616231918335) q[59], q[81];
rzz(0.28706419467926025) q[60], q[81];
rzz(0.5522098541259766) q[61], q[81];
rzz(0.8480010032653809) q[62], q[81];
rzz(0.5860458016395569) q[63], q[81];
rzz(0.5516958236694336) q[64], q[81];
rzz(0.1834787130355835) q[65], q[81];
rzz(0.5259072780609131) q[66], q[81];
rzz(0.10115444660186768) q[67], q[81];
rzz(0.6187025904655457) q[68], q[81];
rzz(0.39406150579452515) q[69], q[81];
rzz(0.7254908084869385) q[70], q[81];
rzz(0.5751828551292419) q[71], q[81];
rzz(0.5344738364219666) q[72], q[81];
rzz(0.04201245307922363) q[73], q[81];
rzz(0.9185656309127808) q[74], q[81];
rzz(0.06732475757598877) q[75], q[81];
rzz(0.03625285625457764) q[76], q[81];
rzz(0.9127851724624634) q[77], q[81];
rzz(0.3245771527290344) q[78], q[81];
rzz(0.3311203122138977) q[79], q[81];
rzz(0.4613441824913025) q[80], q[81];
rzx(0.935337245464325) q[0], q[81];
rzx(0.494179368019104) q[1], q[81];
rzx(0.9497774243354797) q[2], q[81];
rzx(0.5916925668716431) q[3], q[81];
rzx(0.013618290424346924) q[4], q[81];
rzx(0.7697737216949463) q[5], q[81];
rzx(0.19500654935836792) q[6], q[81];
rzx(0.9917600154876709) q[7], q[81];
rzx(0.17024224996566772) q[8], q[81];
rzx(0.8849363923072815) q[9], q[81];
rzx(0.6036857962608337) q[10], q[81];
rzx(0.3224925398826599) q[11], q[81];
rzx(0.6770761609077454) q[12], q[81];
rzx(0.5245132446289062) q[13], q[81];
rzx(0.7720737457275391) q[14], q[81];
rzx(0.8632006049156189) q[15], q[81];
rzx(0.19854849576950073) q[16], q[81];
rzx(0.016380488872528076) q[17], q[81];
rzx(0.38400840759277344) q[18], q[81];
rzx(0.6521525382995605) q[19], q[81];
rzx(0.3394455313682556) q[20], q[81];
rzx(0.2422310709953308) q[21], q[81];
rzx(0.7856813669204712) q[22], q[81];
rzx(0.8400804400444031) q[23], q[81];
rzx(0.8459423780441284) q[24], q[81];
rzx(0.06127351522445679) q[25], q[81];
rzx(0.02514582872390747) q[26], q[81];
rzx(0.06544047594070435) q[27], q[81];
rzx(0.03872370719909668) q[28], q[81];
rzx(0.16790372133255005) q[29], q[81];
rzx(0.05849099159240723) q[30], q[81];
rzx(0.8044359683990479) q[31], q[81];
rzx(0.15880990028381348) q[32], q[81];
rzx(0.5323615074157715) q[33], q[81];
rzx(0.6820365190505981) q[34], q[81];
rzx(0.28577011823654175) q[35], q[81];
rzx(0.5794938206672668) q[36], q[81];
rzx(0.7386534810066223) q[37], q[81];
rzx(0.14319318532943726) q[38], q[81];
rzx(0.7435348629951477) q[39], q[81];
rzx(0.47360366582870483) q[40], q[81];
rzx(0.7298946976661682) q[41], q[81];
rzx(0.7804306149482727) q[42], q[81];
rzx(0.44112247228622437) q[43], q[81];
rzx(0.3921654224395752) q[44], q[81];
rzx(0.4722503423690796) q[45], q[81];
rzx(0.0067632198333740234) q[46], q[81];
rzx(0.11685019731521606) q[47], q[81];
rzx(0.6948805451393127) q[48], q[81];
rzx(0.6095353364944458) q[49], q[81];
rzx(0.2103070616722107) q[50], q[81];
rzx(0.9471233487129211) q[51], q[81];
rzx(0.04869276285171509) q[52], q[81];
rzx(0.1635512113571167) q[53], q[81];
rzx(0.012710750102996826) q[54], q[81];
rzx(0.08516496419906616) q[55], q[81];
rzx(0.8589433431625366) q[56], q[81];
rzx(0.9407163262367249) q[57], q[81];
rzx(0.21288639307022095) q[58], q[81];
rzx(0.2527226209640503) q[59], q[81];
rzx(0.5033541917800903) q[60], q[81];
rzx(0.683894693851471) q[61], q[81];
rzx(0.8167060613632202) q[62], q[81];
rzx(0.0668439269065857) q[63], q[81];
rzx(0.22704440355300903) q[64], q[81];
rzx(0.8656136393547058) q[65], q[81];
rzx(0.28365063667297363) q[66], q[81];
rzx(0.11461323499679565) q[67], q[81];
rzx(0.01581716537475586) q[68], q[81];
rzx(0.19962060451507568) q[69], q[81];
rzx(0.24626761674880981) q[70], q[81];
rzx(0.006206393241882324) q[71], q[81];
rzx(0.36328136920928955) q[72], q[81];
rzx(0.12384504079818726) q[73], q[81];
rzx(0.016666114330291748) q[74], q[81];
rzx(0.30395931005477905) q[75], q[81];
rzx(0.9731080532073975) q[76], q[81];
rzx(0.5016336441040039) q[77], q[81];
rzx(0.10124033689498901) q[78], q[81];
rzx(0.23050379753112793) q[79], q[81];
rzx(0.01921004056930542) q[80], q[81];
h q[0];
