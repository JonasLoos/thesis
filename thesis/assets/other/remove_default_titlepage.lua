if FORMAT:match 'latex' then
    function Meta(m)
        m.title = nil
        return m
    end
end
