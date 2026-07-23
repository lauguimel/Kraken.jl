_build_spec(::Type{MultiphaseSpec}, kw) = throw(phase_stub_error(:multiphase))
_compile_with_spec(::MultiphaseSpec, args...) = throw(phase_stub_error(:multiphase))
_audit_with_spec_type(::Type{MultiphaseSpec}, args...) = throw(phase_stub_error(:multiphase))
