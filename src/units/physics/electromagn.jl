_build_spec(::Type{MHDSpec}, kw) = throw(phase_stub_error(:mhd))
_compile_with_spec(::MHDSpec, args...) = throw(phase_stub_error(:mhd))
_audit_with_spec_type(::Type{MHDSpec}, args...) = throw(phase_stub_error(:mhd))
