OPENQASM 2.0;
include "qelib1.inc";
qreg q[10];
creg c[10];
rz(2.391938874353218) q[3];
cx q[9], q[7];
rz(2.940399839168111) q[6];
rz(5.5010590515644955) q[4];
rz(1.6948062173086589) q[2];
cx q[5], q[1];
rz(6.250394717134311) q[8];
rz(3.304264687777992) q[0];
rz(3.562430685132062) q[4];
rz(3.9818766673381747) q[5];
rz(4.829908326694981) q[9];
cx q[1], q[0];
rz(2.507356598307292) q[3];
rz(3.109023196488791) q[7];
rz(1.8597583916847058) q[2];
rz(0.8668632446070486) q[6];
rz(2.061336023503498) q[8];
rz(4.966317602123409) q[2];
rz(1.5869370998096093) q[7];
rz(0.6913121465009002) q[8];
cx q[4], q[3];
rz(3.159484538023361) q[6];
rz(2.006604352334351) q[9];
rz(4.6738321563360685) q[0];
rz(4.1304474381266285) q[5];
rz(5.264911779466247) q[1];
cx q[0], q[9];
rz(3.4234814847724744) q[5];
rz(5.316602565098891) q[7];
rz(5.152641834665427) q[2];
rz(2.6740259138975424) q[3];
rz(3.2913232660836247) q[8];
cx q[6], q[4];
rz(2.459427715457871) q[1];
rz(4.791248603893335) q[6];
rz(0.13066877052829218) q[5];
rz(0.5975435888449865) q[3];
rz(5.720231398221002) q[9];
rz(4.781766722672305) q[0];
cx q[8], q[1];
cx q[4], q[7];
rz(6.015416979766906) q[2];
rz(0.2980457719163287) q[7];
rz(4.867321872095654) q[6];
rz(2.094420035818164) q[9];
rz(4.316837634059879) q[3];
rz(2.6067385105153207) q[2];
rz(5.3233909529839405) q[5];
rz(4.966590599273841) q[8];
rz(3.9193348490189845) q[1];
cx q[0], q[4];
rz(1.8805049138857415) q[4];
rz(2.185212261701434) q[0];
cx q[8], q[2];
rz(1.6681675206683437) q[5];
rz(0.496752915588005) q[3];
rz(5.775776178103688) q[9];
rz(3.2084883717915464) q[7];
rz(3.3805667147774394) q[6];
rz(3.37019981559306) q[1];
rz(5.202120622841805) q[7];
rz(4.644472678163487) q[8];
cx q[0], q[3];
rz(2.090714793780804) q[2];
cx q[1], q[9];
rz(2.024852553250933) q[6];
rz(3.972420580883204) q[5];
rz(1.8783944270455877) q[4];
rz(2.588827573266969) q[9];
rz(5.464677170091659) q[4];
rz(1.9915655174244917) q[7];
rz(3.4186861429753055) q[2];
rz(0.11160994025719602) q[3];
rz(0.6343318678306554) q[6];
rz(3.0213797507231104) q[1];
rz(0.707947778416336) q[5];
rz(2.0074774764191736) q[8];
rz(2.9257419220269614) q[0];
rz(3.6411826602870336) q[6];
rz(5.205642076976556) q[1];
rz(1.075493248891621) q[5];
rz(2.9946826478703588) q[8];
cx q[2], q[0];
rz(1.425111470056819) q[4];
rz(2.0102480513602736) q[3];
cx q[7], q[9];
rz(1.2478273684985706) q[3];
cx q[8], q[4];
cx q[1], q[5];
rz(4.218508573220149) q[9];
rz(0.8482733180202128) q[6];
rz(3.0160087930431567) q[2];
rz(1.8264056973791707) q[7];
rz(2.8581755800968107) q[0];
cx q[5], q[9];
rz(3.6096705656121073) q[1];
rz(4.219381802858848) q[3];
cx q[8], q[4];
rz(3.1270645787745663) q[7];
cx q[6], q[2];
rz(2.014531353774899) q[0];
rz(0.780649985971022) q[1];
rz(4.08127572995083) q[4];
cx q[6], q[5];
rz(4.081222648678782) q[0];
cx q[8], q[9];
rz(3.7288997304035134) q[3];
rz(3.4921997597147314) q[7];
rz(5.68292934332171) q[2];
rz(5.978616076092748) q[3];
cx q[4], q[2];
rz(1.3102213331372388) q[0];
rz(3.8984619380343384) q[8];
rz(2.883451944601378) q[1];
rz(1.4217087177839298) q[7];
rz(3.6289480601956665) q[5];
rz(1.714684672756237) q[6];
rz(4.282629691437427) q[9];
rz(5.696769714186999) q[1];
rz(4.47219359360426) q[5];
rz(3.4311010154656545) q[0];
rz(3.3392085956378326) q[8];
rz(4.400111011708198) q[4];
cx q[9], q[3];
rz(2.659065148008156) q[6];
rz(4.047278317826741) q[2];
rz(2.9236220943805167) q[7];
rz(0.30062557322245964) q[2];
rz(4.935457731673021) q[0];
cx q[9], q[5];
rz(6.218467939489009) q[1];
rz(6.071958724290096) q[4];
rz(2.7174783228080694) q[8];
cx q[7], q[3];
rz(5.910822356991779) q[6];
rz(5.35299757297472) q[6];
cx q[4], q[5];
rz(1.7611788773720665) q[1];
rz(4.316072913164082) q[2];
cx q[0], q[9];
rz(2.181808334733189) q[8];
cx q[7], q[3];
rz(3.269836749704958) q[1];
rz(1.6394203692746605) q[7];
rz(2.9124736587854296) q[8];
rz(3.132656474470149) q[9];
cx q[5], q[2];
cx q[6], q[0];
cx q[3], q[4];
rz(5.16235246653246) q[4];
cx q[1], q[3];
cx q[8], q[2];
rz(3.614604891664912) q[7];
rz(2.897779676164028) q[0];
rz(2.722419948062154) q[6];
cx q[9], q[5];
rz(0.46615833483329144) q[3];
cx q[9], q[5];
rz(4.92422457141535) q[6];
rz(0.5330452842351908) q[8];
rz(6.276283431444787) q[0];
rz(3.1834769650063164) q[1];
rz(4.432773636521691) q[4];
rz(0.4863060880859504) q[7];
rz(4.72046208723204) q[2];
rz(2.034967064830042) q[3];
cx q[7], q[6];
rz(4.334871516320627) q[9];
rz(5.2770812623858525) q[4];
rz(2.022220810038244) q[2];
rz(1.0072437052277305) q[8];
cx q[5], q[0];
rz(3.9824326403672403) q[1];
rz(3.5193840428594) q[4];
rz(0.6657249533963309) q[1];
rz(1.8149394930947014) q[0];
rz(2.1245057277406882) q[2];
rz(2.9039114780230206) q[9];
cx q[5], q[6];
cx q[7], q[3];
rz(5.7983045089141685) q[8];
rz(0.04346685981761171) q[9];
rz(4.7531257060689045) q[5];
rz(1.8356719206674312) q[1];
rz(4.088877280763616) q[8];
rz(1.9991136256814348) q[7];
rz(5.118344936733253) q[0];
rz(5.916346777078348) q[6];
rz(1.1698920837045081) q[3];
cx q[4], q[2];
rz(1.9920056978607392) q[6];
rz(6.182263583933472) q[9];
rz(6.2453381204450515) q[8];
rz(1.853972856529539) q[3];
rz(5.713154668286203) q[1];
rz(4.026271809475179) q[5];
rz(4.778038743622535) q[7];
rz(2.390471749381772) q[0];
rz(3.056358355680684) q[2];
rz(3.2624342235905424) q[4];
rz(4.032258107278902) q[0];
rz(0.8070675336184875) q[2];
rz(4.59362905460654) q[8];
rz(5.61383834231594) q[1];
rz(0.9583330407852996) q[3];
rz(0.7672180964947313) q[5];
rz(1.0050221558524708) q[9];
cx q[4], q[6];
rz(1.38559288051693) q[7];
rz(1.9540373847880956) q[7];
rz(2.3599420004472447) q[2];
rz(4.331512817325382) q[3];
rz(0.6723770276003445) q[1];
rz(5.302567198878099) q[6];
cx q[4], q[8];
cx q[0], q[9];
rz(1.812254351119041) q[5];
rz(2.2833412262410837) q[7];
rz(1.37748415221459) q[8];
rz(1.320049542485625) q[5];
rz(2.1186658603161796) q[9];
rz(2.519983687212806) q[0];
cx q[3], q[4];
rz(3.367509957396213) q[2];
rz(1.8436698526249593) q[6];
rz(3.3753165109932906) q[1];
rz(0.06141634306512521) q[8];
rz(5.401243109156146) q[5];
rz(1.6873706121257992) q[0];
rz(3.4203528849489118) q[4];
rz(5.38354986308483) q[7];
cx q[3], q[2];
rz(3.1396148540868447) q[9];
rz(5.962385056631352) q[1];
rz(4.9407974343557655) q[6];
rz(2.311075707654079) q[6];
rz(1.0014326270846365) q[3];
rz(2.6710436303501677) q[4];
rz(2.1823026396236194) q[9];
rz(0.922946151902712) q[8];
rz(1.9563551944162076) q[5];
rz(0.5145866856106247) q[7];
rz(4.261191899267032) q[2];
rz(1.6594068008228826) q[1];
rz(5.869563082291108) q[0];
rz(3.1739374624075913) q[1];
rz(1.564062396793358) q[6];
rz(1.1836332520867627) q[5];
rz(6.053103163556587) q[8];
rz(2.233312830102011) q[9];
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
