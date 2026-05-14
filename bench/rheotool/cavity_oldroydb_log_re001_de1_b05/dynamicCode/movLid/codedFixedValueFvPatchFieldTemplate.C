/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) YEAR OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "codedFixedValueFvPatchFieldTemplate.H"
#include "addToRunTimeSelectionTable.H"
#include "fvPatchFieldMapper.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "unitConversion.H"
//{{{ begin codeInclude

//}}} end codeInclude


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

//{{{ begin localCode

//}}} end localCode


// * * * * * * * * * * * * * * * Global Functions  * * * * * * * * * * * * * //

extern "C"
{
    // dynamicCode:
    // SHA1 = 47055371b99582edf0a773f6c3f3d331892f621c
    //
    // unique function name that can be checked if the correct library version
    // has been loaded
    void movLid_47055371b99582edf0a773f6c3f3d331892f621c(bool load)
    {
        if (load)
        {
            // code that can be explicitly executed after loading
        }
        else
        {
            // code that can be explicitly executed before unloading
        }
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

makeRemovablePatchTypeField
(
    fvPatchVectorField,
    movLidFixedValueFvPatchVectorField
);


const char* const movLidFixedValueFvPatchVectorField::SHA1sum =
    "47055371b99582edf0a773f6c3f3d331892f621c";


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

movLidFixedValueFvPatchVectorField::
movLidFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(p, iF)
{
    if (false)
    {
        Info<<"construct movLid sha1: 47055371b99582edf0a773f6c3f3d331892f621c"
            " from patch/DimensionedField\n";
    }
}


movLidFixedValueFvPatchVectorField::
movLidFixedValueFvPatchVectorField
(
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const dictionary& dict
)
:
    fixedValueFvPatchField<vector>(p, iF, dict)
{
    if (false)
    {
        Info<<"construct movLid sha1: 47055371b99582edf0a773f6c3f3d331892f621c"
            " from patch/dictionary\n";
    }
}


movLidFixedValueFvPatchVectorField::
movLidFixedValueFvPatchVectorField
(
    const movLidFixedValueFvPatchVectorField& ptf,
    const fvPatch& p,
    const DimensionedField<vector, volMesh>& iF,
    const fvPatchFieldMapper& mapper
)
:
    fixedValueFvPatchField<vector>(ptf, p, iF, mapper)
{
    if (false)
    {
        Info<<"construct movLid sha1: 47055371b99582edf0a773f6c3f3d331892f621c"
            " from patch/DimensionedField/mapper\n";
    }
}


movLidFixedValueFvPatchVectorField::
movLidFixedValueFvPatchVectorField
(
    const movLidFixedValueFvPatchVectorField& ptf,
    const DimensionedField<vector, volMesh>& iF
)
:
    fixedValueFvPatchField<vector>(ptf, iF)
{
    if (false)
    {
        Info<<"construct movLid sha1: 47055371b99582edf0a773f6c3f3d331892f621c "
            "as copy/DimensionedField\n";
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

movLidFixedValueFvPatchVectorField::
~movLidFixedValueFvPatchVectorField()
{
    if (false)
    {
        Info<<"destroy movLid sha1: 47055371b99582edf0a773f6c3f3d331892f621c\n";
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void movLidFixedValueFvPatchVectorField::updateCoeffs()
{
    if (this->updated())
    {
        return;
    }

    if (false)
    {
        Info<<"updateCoeffs movLid sha1: 47055371b99582edf0a773f6c3f3d331892f621c\n";
    }

//{{{ begin code
    #line 30 "//data/0/U/boundaryField/movingLid"
const scalar& t = this->db().time().timeOutputValue() ;
 
          const vectorField& x = patch().Cf();

          operator==(  
                       vector(1., 0., 0.)  
                     * 8.0 * (1.0 + tanh( 8.0 * (t - 0.5) ) ) 
                     * pow( x.component(0), 2.0 ) * pow( ( 1 - x.component(0) ), 2.0  )
                    );
          
          fixedValueFvPatchVectorField::updateCoeffs();
//}}} end code

    this->fixedValueFvPatchField<vector>::updateCoeffs();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// ************************************************************************* //

