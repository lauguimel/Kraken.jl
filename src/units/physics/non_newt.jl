_build_spec(::Type{PowerLawSpec}, kw) = throw(phase_stub_error(:power_law))
_compile_with_spec(::PowerLawSpec, args...) = throw(phase_stub_error(:power_law))
_audit_with_spec_type(::Type{PowerLawSpec}, args...) = throw(phase_stub_error(:power_law))
