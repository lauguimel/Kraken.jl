using Kraken, Metal, KernelAbstractions, Printf
backend=MetalBackend(); T=Float32
for R in (10, 20)
    Nx=30R; Ny=4R; cx=15R; cy=2R; u_mean=T(0.02); u_max=T(1.5)*u_mean; ν=T(0.02R)
    steps = R == 10 ? 20_000 : 40_000; avg = steps ÷ 4
    qh, sh = Kraken.precompute_q_wall_cylinder(Nx, Ny, cx, cy, R; FT=T)
    u_prof_h = [T(4)*u_max*T(j-1)*T(Ny-j)/T(Ny-1)^2 for j in 1:Ny]
    f_h=zeros(T,Nx,Ny,9)
    for j in 1:Ny, i in 1:Nx, q in 1:9
        f_h[i,j,q]=Kraken.equilibrium(D2Q9(), one(T), u_prof_h[j], zero(T), q)
    end
    q_wall=KernelAbstractions.allocate(backend,T,Nx,Ny,9); copyto!(q_wall,qh)
    is_solid=KernelAbstractions.allocate(backend,Bool,Nx,Ny); copyto!(is_solid,sh)
    uw_x=KernelAbstractions.zeros(backend,T,Nx,Ny,9); uw_y=KernelAbstractions.zeros(backend,T,Nx,Ny,9)
    f_in=KernelAbstractions.allocate(backend,T,Nx,Ny,9); copyto!(f_in,f_h)
    f_out=KernelAbstractions.zeros(backend,T,Nx,Ny,9)
    ρ=KernelAbstractions.zeros(backend,T,Nx,Ny); fill!(ρ,one(T))
    ux=KernelAbstractions.zeros(backend,T,Nx,Ny); uy=KernelAbstractions.zeros(backend,T,Nx,Ny)
    u_profile=KernelAbstractions.allocate(backend,T,Ny); copyto!(u_profile,u_prof_h)
    meiFx=0.0; meaFx=0.0; n=0
    @printf("\nR=%d steps=%d avg=%d\n", R, steps, avg); flush(stdout)
    for step in 1:steps
        Kraken.fused_trt_libb_v2_step!(f_out,f_in,ρ,ux,uy,is_solid,q_wall,uw_x,uw_y,Nx,Ny,ν)
        Kraken.rebuild_inlet_outlet_libb_2d!(f_out,f_in,u_profile,one(T),ν,Nx,Ny)
        if step > steps-avg
            dmei=Kraken.compute_drag_libb_mei_2d(f_out,q_wall,uw_x,uw_y,Nx,Ny)
            dmea=Kraken.compute_drag_mea_2d(f_in,f_out,is_solid,Nx,Ny)
            meiFx += dmei.Fx; meaFx += dmea.Fx; n += 1
        end
        f_in,f_out=f_out,f_in
    end
    KernelAbstractions.synchronize(backend)
    D=2R; uref=Float64(u_mean)
    cdmei=2*(meiFx/n)/(uref^2*D)
    cdmea=2*(meaFx/n)/(uref^2*D)
    @printf("Cd_Mei=%.6f Cd_MEA_stair=%.6f ratio=%.6f\n", cdmei, cdmea, cdmea/cdmei); flush(stdout)
end
