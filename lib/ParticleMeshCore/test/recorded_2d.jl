# The 2D Reference digests recorded on a commit (WP-037), checked by T-266 (engine2d_digests) and
# T-303 (flux2d_digests), as T-260 checks the N = 1 bits against N1_RECORDED. Needs digest.jl,
# engine2d_cases.jl and flux2d_cases.jl. The lines are `"kind|type|case|digest"`: the launch shape
# is dropped, since T-266 and T-303 already require every launch shape to give the same bits.
# Computed on 86bcf55 (main before WP-037), on the CPU backend with Julia 1.13 at 1 and 4 threads
# (identical). A change to these bits is a change of a Reference result (A4): rerun
# `recorded_2d_lines` and replace the lines only when the change is intended, and say so in the PR.

"`kind|type|case|digest` of digest lines `kind|type|case|workgroupsize|digest`, one per case."
recorded_2d_lines(lines) = unique([let f = split(l, '|'); join((f[1:3]..., f[5]), '|') end for l in lines])

const E2D_RECORDED = String[
    "sort2d|Float32|extras|15543817846396621455",
    "deposit2d|Float32|1,Node,Node,qw|17683428208705412539",
    "deposit2d|Float32|1,Node,Node,qwuy|10781171832613280278",
    "gather2d|Float32|1,Node,Node|6198051181525136308",
    "deposit2d|Float32|1,Center,Node,qw|11996613351757936745",
    "deposit2d|Float32|1,Center,Node,qwuy|14863886815925509345",
    "deposit2d|Float32|2,Center,Center,qw|6033423277655256889",
    "deposit2d|Float32|2,Center,Center,qwuy|14042158852234014432",
    "gather2d|Float32|2,Center,Center|1009973278119388540",
    "deposit2d|Float32|2,Center,Node,qw|15124772504272814440",
    "deposit2d|Float32|2,Center,Node,qwuy|4588399388274989273",
    "fill2d|Float32|9x4|6485492503439700597",
    "fold2d|Float32|9x4|40734728018716922",
    "load2d|Float32|ppc=2x3|13088088606221365582",
    "sort2d|Float64|extras|12156883673378848697",
    "deposit2d|Float64|1,Node,Node,qw|15634287146082216116",
    "deposit2d|Float64|1,Node,Node,qwuy|16188244835812999264",
    "gather2d|Float64|1,Node,Node|1482327362590600939",
    "deposit2d|Float64|1,Center,Node,qw|17989555346631472774",
    "deposit2d|Float64|1,Center,Node,qwuy|8802786807452504474",
    "deposit2d|Float64|2,Center,Center,qw|10247293542371762926",
    "deposit2d|Float64|2,Center,Center,qwuy|16108184129641111741",
    "gather2d|Float64|2,Center,Center|8789137941361927838",
    "deposit2d|Float64|2,Center,Node,qw|16864153857190077042",
    "deposit2d|Float64|2,Center,Node,qwuy|8825628543626757974",
    "fill2d|Float64|9x4|7629326602720750849",
    "fold2d|Float64|9x4|13214075647978165953",
    "load2d|Float64|ppc=2x3|11107899924285759324",
    "sort2d|Dual{Float32,3}|extras|1559576997544743955",
    "deposit2d|Dual{Float32,3}|1,Node,Node,qw|13825327665839195361",
    "deposit2d|Dual{Float32,3}|1,Node,Node,qwuy|14432473154432260456",
    "gather2d|Dual{Float32,3}|1,Node,Node|11963303085289519855",
    "deposit2d|Dual{Float32,3}|1,Center,Node,qw|17006825278466049045",
    "deposit2d|Dual{Float32,3}|1,Center,Node,qwuy|10645627902574188545",
    "deposit2d|Dual{Float32,3}|2,Center,Center,qw|6258135950967033608",
    "deposit2d|Dual{Float32,3}|2,Center,Center,qwuy|5910202189410886186",
    "gather2d|Dual{Float32,3}|2,Center,Center|10809536326392098092",
    "deposit2d|Dual{Float32,3}|2,Center,Node,qw|10214139921277133594",
    "deposit2d|Dual{Float32,3}|2,Center,Node,qwuy|10964343316795342244",
    "fill2d|Dual{Float32,3}|9x4|8534630521971183493",
    "fold2d|Dual{Float32,3}|9x4|12030501656798386926",
    "load2d|Dual{Float32,3}|ppc=2x3|5259342032513399083",
]

const F2D_RECORDED = String[
    "flux2d|Float32|1,Node,Node|18259945354132649916",
    "flux2d|Float32|1,Center,Center|5499945992414612418",
    "flux2d|Float32|2,Node,Node|15991085678762985927",
    "flux2d|Float32|2,Center,Center|2417744543097394108",
    "flux2d|Float64|1,Node,Node|1800987136620872041",
    "flux2d|Float64|1,Center,Center|10095521823029044306",
    "flux2d|Float64|2,Node,Node|14012135646213642108",
    "flux2d|Float64|2,Center,Center|5609872212990875902",
    "flux2d|Dual{Float32,3}|1,Node,Node|4314086691216750655",
    "flux2d|Dual{Float32,3}|1,Center,Center|4244957048672191451",
    "flux2d|Dual{Float32,3}|2,Node,Node|12456855201821418007",
    "flux2d|Dual{Float32,3}|2,Center,Center|2307368713551168192",
]
