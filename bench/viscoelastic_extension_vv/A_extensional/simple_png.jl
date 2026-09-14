function _m44_u32be(x::Integer)
    return UInt8[(x >> 24) & 0xff, (x >> 16) & 0xff, (x >> 8) & 0xff, x & 0xff]
end

function _m44_crc32(bytes::Vector{UInt8})
    crc = UInt32(0xffffffff)
    for b in bytes
        crc = xor(crc, UInt32(b))
        for _ in 1:8
            mask = ifelse((crc & UInt32(1)) == UInt32(1), UInt32(0xedb88320), UInt32(0))
            crc = xor(crc >> 1, mask)
        end
    end
    return xor(crc, UInt32(0xffffffff))
end

function _m44_adler32(bytes::Vector{UInt8})
    a = UInt32(1)
    b = UInt32(0)
    modv = UInt32(65521)
    for x in bytes
        a = (a + UInt32(x)) % modv
        b = (b + a) % modv
    end
    return (b << 16) | a
end

function _m44_zlib_store(data::Vector{UInt8})
    out = UInt8[0x78, 0x01]
    pos = 1
    while pos <= length(data)
        len = min(65535, length(data) - pos + 1)
        final = pos + len - 1 == length(data)
        push!(out, final ? UInt8(0x01) : UInt8(0x00))
        n = UInt16(len)
        nn = ~n
        append!(out, UInt8[n & 0xff, (n >> 8) & 0xff, nn & 0xff, (nn >> 8) & 0xff])
        append!(out, @view data[pos:pos + len - 1])
        pos += len
    end
    append!(out, _m44_u32be(_m44_adler32(data)))
    return out
end

function _m44_png_chunk(io, typ::String, data::Vector{UInt8})
    typebytes = Vector{UInt8}(codeunits(typ))
    write(io, _m44_u32be(length(data)))
    write(io, typebytes)
    write(io, data)
    write(io, _m44_u32be(_m44_crc32(vcat(typebytes, data))))
end

function m44_save_rgb_png(path::AbstractString, img::Array{UInt8,3})
    h, w, c = size(img)
    c == 3 || throw(ArgumentError("expected RGB image with third dimension 3"))
    raw = UInt8[]
    sizehint!(raw, h * (1 + 3w))
    for y in 1:h
        push!(raw, 0x00)
        for x in 1:w
            push!(raw, img[y, x, 1]); push!(raw, img[y, x, 2]); push!(raw, img[y, x, 3])
        end
    end
    ihdr = UInt8[]
    append!(ihdr, _m44_u32be(w)); append!(ihdr, _m44_u32be(h))
    append!(ihdr, UInt8[0x08, 0x02, 0x00, 0x00, 0x00])
    open(path, "w") do io
        write(io, UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a])
        _m44_png_chunk(io, "IHDR", ihdr)
        _m44_png_chunk(io, "IDAT", _m44_zlib_store(raw))
        _m44_png_chunk(io, "IEND", UInt8[])
    end
    return path
end

function m44_canvas(w::Integer, h::Integer; color=(0xff, 0xff, 0xff))
    img = Array{UInt8}(undef, h, w, 3)
    img[:, :, 1] .= UInt8(color[1])
    img[:, :, 2] .= UInt8(color[2])
    img[:, :, 3] .= UInt8(color[3])
    return img
end

function m44_setpixel!(img, x::Integer, y::Integer, color)
    h, w, _ = size(img)
    if 1 <= x <= w && 1 <= y <= h
        img[y, x, 1] = UInt8(color[1])
        img[y, x, 2] = UInt8(color[2])
        img[y, x, 3] = UInt8(color[3])
    end
    return nothing
end

function m44_line!(img, x1, y1, x2, y2, color; width::Integer=1)
    steps = max(1, ceil(Int, max(abs(x2 - x1), abs(y2 - y1))))
    r = max(0, div(width, 2))
    for k in 0:steps
        a = k / steps
        x = round(Int, (1 - a) * x1 + a * x2)
        y = round(Int, (1 - a) * y1 + a * y2)
        for dy in -r:r, dx in -r:r
            m44_setpixel!(img, x + dx, y + dy, color)
        end
    end
    return nothing
end

function m44_rect!(img, x1, y1, x2, y2, color)
    m44_line!(img, x1, y1, x2, y1, color)
    m44_line!(img, x2, y1, x2, y2, color)
    m44_line!(img, x2, y2, x1, y2, color)
    m44_line!(img, x1, y2, x1, y1, color)
    return nothing
end
